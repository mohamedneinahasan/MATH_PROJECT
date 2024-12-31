import argparse
import os
import logging
import json
import math
from tqdm import tqdm
from utils import *
from collections import Counter
import time

import torch
from torch.utils.data import DataLoader
from accelerate.logging import get_logger
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    PeftModel,
    prepare_model_for_kbit_training,
)

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test Evaluation for Math Problem Solving"
    )

    parser.add_argument(
        "--hf_auth_token",
        type=str,
        required=True,
        help="Huggingface authorization token",
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Run to evaluate",
    )
    parser.add_argument(
        "--sc_batch_size",
        type=int,
        default=1,
        help="Batch size for self consistency evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for greedy evalaution"
    )

    # Generation
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=False,
        help="Whether or not to use sampling",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams for beam serach, 1 means no beam search",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The value used to modulate the next token probabilities",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        default=False,
        help="Controls the stopping condition for beam-based methods, like beam-search",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="The number of independently computed returned sequences for each element in the batch",
    )

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    with open(f"{args.run}/training_args.json") as f:
        training_args = dotdict(json.load(f))

    # Initialize accelerator
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    output_dir = f"{args.run}/evaluation/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    # Args jsonl to store all args
    with open(
        f"{output_dir}/generation_args.json", "w", encoding="utf-8"
    ) as write_file:
        json.dump(dict(vars(args)), write_file, ensure_ascii=False, indent=4)

    accelerator.wait_for_everyone()

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        f"{args.run}/tokenizer",
        local_files_only=True,
        cache_dir="models",
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=training_args.load_in_4bit,
        bnb_4bit_quant_type=training_args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=training_args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=training_args.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        training_args.model,
        quantization_config=bnb_config,
        pad_token_id=tokenizer.eos_token_id,  # https://github.com/microsoft/DeepSpeedExamples/issues/490
        cache_dir="models",
        use_auth_token=args.hf_auth_token,
    )

    model = prepare_model_for_kbit_training(model)

    model = PeftModel.from_pretrained(model, f"{args.run}/peft_adapter")
    model.print_trainable_parameters()
    accelerator.print(f"Loaded adapter from {f'{args.run}/peft_adapter'}")

    # Setting generation config for validation
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.generation_config.max_new_tokens = training_args.max_length
    model.generation_config.num_beams = args.num_beams
    model.generation_config.early_stopping = args.early_stopping
    model.generation_config.do_sample = args.do_sample
    model.generation_config.temperature = args.temperature
    model.generation_config.top_p = args.top_p
    model.generation_config._from_model_config = False

    logger.info("Loading dataset")
    test_dataset = load_from_disk(f"data/{training_args.dataset_name}")["test"]

    max_length = training_args.max_length
    with accelerator.main_process_first():
        test_tokenized_dataset = test_dataset.map(
            lambda x: tokenize_data(
                x, tokenizer=tokenizer, is_train=False, max_length=max_length
            ),
            batched=True,
            load_from_cache_file=False,
            num_proc=training_args.num_proc,
            remove_columns=test_dataset.column_names,
        )

    if args.num_return_sequences > 1:
        batch_size = args.sc_batch_size
    else:
        batch_size = args.batch_size

    collator = PaddedDataCollator(tokenizer=tokenizer)
    test_dataloader = DataLoader(
        test_tokenized_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collator,
    )

    accelerator.print(tokenizer.decode(next(iter(test_dataloader))["input_ids"][0]))

    # Set up runtime for code execution
    runtime = GenericRuntime()

    all_metadata = test_dataset

    logger.info("Preparing model")
    model = accelerator.prepare(model)

    logger.info("Preparing dataset")
    test_dataloader = accelerator.prepare(test_dataloader)

    total_batch_size = batch_size * accelerator.num_processes

    logger.info("***** Running Test Evaluation *****")
    logger.info(f"  Run = {args.run}")
    logger.info(f"  Num test examples = {len(test_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(
        f"  Total test batch size (w. parallel, distributed) = {total_batch_size}"
    )

    model.eval()

    logger.info("Start evaluation")
    module = accelerator.unwrap_model(model)

    predictions = []
    with torch.no_grad():
        for index, feature in tqdm(
            enumerate(test_dataloader),
            disable=not accelerator.is_local_main_process,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            desc="--test",
            total=len(test_dataloader),
        ):
            generated_ids = module.generate(
                input_ids=feature["input_ids"],
                attention_mask=feature["attention_mask"],
                num_return_sequences=args.num_return_sequences,
                return_dict_in_generate=True,
            ).sequences

            generated_ids = generated_ids[
                :, feature["input_ids"].size(1) :
            ].contiguous()
            generated_ids = accelerator.pad_across_processes(
                generated_ids, dim=1, pad_index=tokenizer.eos_token_id
            )
            generated_ids = accelerator.gather_for_metrics(generated_ids)
            prediction = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            sorted_prediction = [
                prediction[
                    i * args.num_return_sequences : (i + 1) * args.num_return_sequences
                ]
                for i in range(len(prediction))
            ]
            predictions.extend(sorted_prediction)

        if accelerator.is_main_process:
            correct = 0

            # Run execution and compare to ground truth
            result_counter = Counter()
            all_result = []

            for samples, metadata in zip(predictions, all_metadata):
                all_code = []
                all_numeric_ans = []

                for sample in samples:
                    try:
                        code, ans, error = runtime.run_code(
                            code_gen=sample, answer_expr="solution()"
                        )
                    except:
                        code = ""
                        ans = None
                    try:
                        numeric_ans = float(ans)
                    except:
                        numeric_ans = None

                    if numeric_ans is not None:
                        result_counter.update([numeric_ans])

                    all_code.append(code)
                    all_numeric_ans.append(numeric_ans)

                if len(result_counter) > 0:
                    final_ans = result_counter.most_common(1)[0][0]
                else:
                    final_ans = None

                score = 0
                if (
                    final_ans is not None
                    and math.fabs(final_ans - float(metadata["answer"])) < 1e-2
                ):
                    correct += 1
                    score = 1

                all_result.append(
                    {
                        "question": metadata["question"],
                        "answer": metadata["answer"],
                        "score": score,
                        "final_prediction": final_ans,
                        "predicted_answer": all_numeric_ans,
                        "code": all_code,
                        "generation": samples,
                    }
                )

                result_counter.clear()

            # Store outputs
            with open(
                f"{output_dir}/test_outputs.json", "a", encoding="utf-8"
            ) as write_file:
                json.dump(all_result, write_file, ensure_ascii=False, indent=4)
                write_file.write("\n")

            test_accuracy = correct / len(test_tokenized_dataset)
            accelerator.print(f"Test Accuracy = {test_accuracy}")

            # Store result
            with open(f"{output_dir}/result.jsonl", "a") as f:
                f.write(json.dumps({"Test Accuracy": test_accuracy}))


if __name__ == "__main__":
    main()
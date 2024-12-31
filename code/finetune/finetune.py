import os
import time
import json
import math
import logging
import argparse
from utils import *
from tqdm import tqdm

from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    AdamW,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine Tuning for Math Problem Solving")

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k_ucp",
        help="Dataset name for finetuning",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="codellama/CodeLlama-7b-hf",
        help="The model to finetune",
    )
    parser.add_argument(
        "--hf_auth_token",
        type=str,
        required=True,
        help="Huggingface authorization token",
    )

    # Training
    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="How many examples to use for training, -1 means all examples",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Max generation length"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=20,
        help="Total number of training epochs to perform",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning Rate of optimizer",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-8,
        help="Epsilon value for the AdamW optimizer",
    )
    parser.add_argument(
        "--warmup_step", type=float, default=-1, help="Scheduler warm up steps"
    )
    parser.add_argument(
        "--max_grad_norm", type=float, default=0.3, help="The maximum gradient norm"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use"
    )
    parser.add_argument("--num_proc", type=int, default=8, help="Dataset map num proc")

    # lora config
    parser.add_argument("--lora_r", type=int, default=64, help="r value for lora")
    parser.add_argument(
        "--lora_alpha", type=int, default=16, help="Alpha value for lora"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="Dropout value for lora"
    )
    parser.add_argument("--bias", type=str, default="none", help="Bias value for lora")

    # bitsandbytes config
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=False,
        help="Activate 4bit precision base model loading",
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        help="Quantization type fp4 or nf4",
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        default="bfloat16",
        help="Compute dtype for 4bit base models",
    )
    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        default=False,
        help="Activate nested quantization for 4bit base models",
    )

    args = parser.parse_args()
    return args


def main():
    # Parse arguments
    args = parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set training seed if specified
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation
    output_dir = (
        "saved/"
        + f'{time.strftime("%Y%m%d-%H%M%S")}_{args.model.split("/")[-1]}_{args.num_examples}_{args.dataset_name}'
    )
    if accelerator.is_main_process:
        os.makedirs("saved", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Store all args
        with open(
            f"{output_dir}/training_args.json", "w", encoding="utf-8"
        ) as write_file:
            json.dump(dict(vars(args)), write_file, ensure_ascii=False, indent=4)

        accelerator.project_configuration.automatic_checkpoint_naming = False

    accelerator.wait_for_everyone()

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, cache_dir="models", use_auth_token=args.hf_auth_token
    )

    # Load model
    logger.info("Loading model...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        pad_token_id=tokenizer.eos_token_id,
        cache_dir="models",
        use_auth_token=args.hf_auth_token,
    )
    model = prepare_model_for_kbit_training(model)

    modules = find_all_linear_names(args, model)
    # Initialize QLoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_from_disk(f"data/{args.dataset_name}")

    # Preprocess dataset
    train_dataset = dataset["train"]

    if args.num_examples != -1:
        train_dataset = train_dataset.select(list(range(args.num_examples)))

    with accelerator.main_process_first():
        train_tokenized_dataset = train_dataset.map(
            lambda x: tokenize_data(
                x, tokenizer=tokenizer, is_train=True, max_length=args.max_length
            ),
            batched=True,
            num_proc=args.num_proc,
            load_from_cache_file=False,
            remove_columns=train_dataset.column_names,
        )
    accelerator.wait_for_everyone()

    # Prepare dataloader
    collator = PaddedDataCollator(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_tokenized_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collator,
    )

    accelerator.print(tokenizer.decode(next(iter(train_dataloader))["input_ids"][0]))

    # Calculating total steps
    total_steps = math.ceil(len(train_dataloader)) * args.num_train_epochs

    # Initialize optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.epsilon,
        no_deprecation_warning=True,
    )
    warmup_step = args.warmup_step if args.warmup_step >= 0 else int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_step,
        num_training_steps=total_steps,
    )

    # Prepare for training
    logger.info("Preparing model...")
    model = accelerator.prepare(model)
    optimizer = accelerator.prepare(optimizer)
    scheduler = accelerator.prepare(scheduler)

    logger.info("Preparing dataset...")
    train_dataloader = accelerator.prepare(train_dataloader)

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes

    logger.info("***** Running training *****")
    logger.info(f"  Model = {args.model}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device (training) = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed) = {total_batch_size}"
    )
    logger.info(f"  Total optimization steps = {total_steps}")
    logger.info(f"  Saved folder path = {output_dir}")

    # Training loop
    for epoch in range(1, args.num_train_epochs + 1):

        # Train model
        total_loss = 0
        model.train()

        for iteration, feature in tqdm(
            enumerate(train_dataloader, start=1),
            disable=not accelerator.is_local_main_process,
            bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
            desc="--training batch",
            total=len(train_dataloader),
        ):
            optimizer.zero_grad()
            outputs = model(**feature)
            loss = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            total_loss += loss.item()
            optimizer.step()
            scheduler.step()

        result = {}
        result["epoch"] = epoch
        result["train_loss"] = total_loss / len(train_dataloader)

        result_string = f"Epoch: {epoch}, Train Loss: {result['train_loss']}\n"
        accelerator.print(result_string)

        if accelerator.is_main_process:
            # Store result
            with open(f"{output_dir}/result.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
        accelerator.wait_for_everyone()

    # Saving adapters
    peft_adapter_output_dir = os.path.join(output_dir, "peft_adapter")
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(peft_adapter_output_dir)

    # Saving tokenizer
    tokenizer_output_dir = os.path.join(output_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_output_dir)
    accelerator.print(f"Saving to {output_dir} complete...")


if __name__ == "__main__":
    main()

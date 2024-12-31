import os
import json
import argparse
from pathlib import Path
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import torch

def setup_args():
    parser = argparse.ArgumentParser(description='Evaluate math models on test datasets')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where output JSON files will be saved'
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='Path to the JSON dataset file containing questions and answers'
    )
    parser.add_argument(
        '--hf_token',
        type=str,
        required=True,
        help='Hugging Face API token'
    )
    return parser.parse_args()

def load_dataset(dataset_path):
    """Load dataset from a JSON file."""
    with open(dataset_path, 'r') as f:
        return json.load(f)

def save_results(results, output_dir, model_name):
    """Save results to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'CoT_{model_name}.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def main():
    args = setup_args()

    # Authenticate with Hugging Face
    login(token=args.hf_token)

    # Load dataset
    dataset = load_dataset(args.dataset_path)

    # Define model IDs to evaluate
    model_ids = [
        "meta-llama/Meta-Llama-3.1-8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "mistralai/Mathstral-7B-v0.1"
    ]

    device_map = "balanced" if torch.cuda.device_count() > 1 else "auto"

    # Iterate over each model
    for model_id in model_ids:
        print(f"\nEvaluating model: {model_id}")
        model_results = {}
        correct_answers = 0
        total_questions = 0

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True),
            torch_dtype=torch.bfloat16,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Initialize pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )

        for item in dataset:
            question = item["algebraic_word_problem"]
            reference_answer = item["reference_answer"]

            # Generate prediction
            input_prompt = f"Question: {question}\nAnswer:"
            outputs = generator(input_prompt, max_new_tokens=50, do_sample=False)
            predicted_answer = outputs[0]["generated_text"].strip()

            # Compare predicted answer with reference answer
            if reference_answer in predicted_answer:
                correct_answers += 1

            total_questions += 1

        # Calculate accuracy
        accuracy = correct_answers / total_questions if total_questions > 0 else 0
        model_results["accuracy"] = accuracy

        # Save results
        save_results(model_results, args.output_dir, model_id.replace("/", "_"))

        # Clear GPU cache
        torch.cuda.empty_cache()

    print("Evaluation completed.")

if __name__ == "__main__":
    main()

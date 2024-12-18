import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from fuzzywuzzy import fuzz
import time
import multiprocessing

# Load the model and tokenizer (shared across GPUs)
model_name = "mistralai/Mathstral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to solve math problems (modified to accept specific device)
def solve_math_problem(problem_text, num_attempts=1, model=None, tokenizer=None, device=None):
    prompt = f"Solve the following math problem: {problem_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    outputs = []
    for _ in range(num_attempts):
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        outputs.append(output_text)
    return outputs

# Function to compute Exact Match and Fuzzy Score
def exact_match(predicted_answer, reference_answer):
    return int(predicted_answer.strip() == reference_answer.strip())

def fuzzy_score(predicted_answer, reference_answer):
    return fuzz.ratio(predicted_answer.strip(), reference_answer.strip())

# Evaluation function for a subset of the dataset
def evaluate_model_subset(dataset, device_id):
    # Load model to the specific GPU
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device_id)
    total_questions = len(dataset)
    maj1 = rr_all = rr_top_8 = 0
    fuzzy_scores = []
    predicted_answers_list = []
    total_time_taken = 0

    for idx, row in dataset.iterrows():
        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Start time tracking for each problem
        start_time = time.time()

        # Get predictions for multiple attempts on the specified GPU
        predicted_answers = solve_math_problem(problem_text, num_attempts=64, model=model, tokenizer=tokenizer, device=device_id)

        # End time tracking for each problem
        end_time = time.time()
        time_taken = end_time - start_time
        total_time_taken += time_taken

        # Calculate Maj1@64: at least 64% of predictions correct
        correct_answers_count = sum(exact_match(answer, reference_answer) for answer in predicted_answers)
        if correct_answers_count >= 41:
            maj1 += 1

        # Calculate RR.All and RR.Top-8
        if any(exact_match(answer, reference_answer) for answer in predicted_answers):
            rr_all += 1
        if any(exact_match(answer, reference_answer) for answer in predicted_answers[:8]):
            rr_top_8 += 1

        # Calculate fuzzy score for the first prediction
        fuzzy = fuzzy_score(predicted_answers[0], reference_answer)
        fuzzy_scores.append(fuzzy)

        # Append results for individual analysis
        row['predicted_answer'] = predicted_answers[0]
        row['fuzzy_score'] = fuzzy
        predicted_answers_list.append(row)

    # Calculate metrics for this subset
    maj1_rate = maj1 / total_questions * 100
    rr_all_rate = rr_all / total_questions * 100
    rr_top_8_rate = rr_top_8 / total_questions * 100
    avg_fuzzy_score = sum(fuzzy_scores) / total_questions
    avg_time_taken = total_time_taken / total_questions

    # Save results to a file specific to this GPU's subset
    output_df = pd.DataFrame(predicted_answers_list)
    output_df.to_csv(f'predicted_answers_with_metrics_gpu_{device_id}.csv', index=False)

    return maj1_rate, rr_all_rate, rr_top_8_rate, avg_fuzzy_score, avg_time_taken

# Main function to parallelize across GPUs
def main_evaluation(dataset_path):
    # Load dataset
    dataset = pd.read_json(dataset_path)
    
    # Split dataset into two halves
    half = len(dataset) // 2
    datasets = [dataset.iloc[:half].reset_index(drop=True), dataset.iloc[half:].reset_index(drop=True)]

    # Define devices for each split
    devices = ["cuda:0", "cuda:1"]

    # Start multiprocessing with each subset assigned to a separate GPU
    with multiprocessing.Pool(2) as pool:
        results = pool.starmap(evaluate_model_subset, zip(datasets, devices))

    # Calculate combined metrics
    maj1_rate = sum(result[0] for result in results) / 2
    rr_all_rate = sum(result[1] for result in results) / 2
    rr_top_8_rate = sum(result[2] for result in results) / 2
    avg_fuzzy_score = sum(result[3] for result in results) / 2
    avg_time_taken = sum(result[4] for result in results) / 2

    print(f"Maj1@64 Accuracy: {maj1_rate:.2f}%")
    print(f"RR.All Accuracy: {rr_all_rate:.2f}%")
    print(f"RR.Top-8 Accuracy: {rr_top_8_rate:.2f}%")
    print(f"Average Fuzzy Matching Score: {avg_fuzzy_score:.2f}")
    print(f"Average Time Taken per Problem: {avg_time_taken:.2f} seconds")

# Run the evaluation with the dataset path
main_evaluation('scoredataset.json')

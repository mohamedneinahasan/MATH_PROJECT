import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import time
import multiprocessing

# Load the model and tokenizer
model_name = "mistralai/Mathstral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Split the dataset for two GPUs
def split_dataset(dataset, num_splits=2):
    split_size = len(dataset) // num_splits
    return [dataset.iloc[i*split_size: (i+1)*split_size] for i in range(num_splits)]

# Function to solve math problems on a specific GPU
def solve_math_problem(model, problem_text, device, num_attempts=16):
    prompt = f"Solve the following math problem: {problem_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=150,
            num_return_sequences=num_attempts,  # Generate multiple answers
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    outputs = [tokenizer.decode(output_id, skip_special_tokens=True).strip() for output_id in output_ids]
    return outputs

# Maj1@16 Evaluation function
def maj1_at_16(predicted_answers, reference_answer):
    return int(reference_answer.strip() in [answer.strip() for answer in predicted_answers])

# Evaluation function on a specific GPU
def evaluate_on_gpu(gpu_id, dataset_split):
    # Set device
    device = torch.device(f"cuda:{gpu_id}")
    
    # Load model on the specific GPU
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    predicted_answers_list = []
    maj1_at_16_count = 0
    total_time_taken = 0

    for idx, row in dataset_split.iterrows():
        print(f"[GPU {gpu_id}] Solving problem ID {idx + 1}/{len(dataset_split)}...")

        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Track the start time for solving the problem
        start_time = time.time()

        # Generate a list of predicted answers (16 attempts)
        predicted_answers = solve_math_problem(model, problem_text, device, num_attempts=16)

        # Track the end time after solving the problem
        end_time = time.time()
        time_taken = end_time - start_time
        total_time_taken += time_taken

        # Check Maj1@16
        maj1_at_16_count += maj1_at_16(predicted_answers, reference_answer)

        # Store results for analysis
        row['predicted_answers'] = predicted_answers
        row['maj1_at_16'] = maj1_at_16(predicted_answers, reference_answer)
        predicted_answers_list.append(row)

    # Calculate Maj1@16 score
    maj1_at_16_score = maj1_at_16_count / len(dataset_split) * 100
    avg_time_taken = total_time_taken / len(dataset_split)

    return predicted_answers_list, maj1_at_16_score, avg_time_taken

# Run evaluation on two GPUs and aggregate results
def main_evaluation(dataset):
    # Split dataset for each GPU
    dataset_splits = split_dataset(dataset, num_splits=2)

    # Start parallel processing on two GPUs
    with multiprocessing.Pool(2) as pool:
        results = pool.starmap(evaluate_on_gpu, [(0, dataset_splits[0]), (1, dataset_splits[1])])

    # Aggregate results from both GPUs
    all_predicted_answers = []
    total_maj1_at_16_count = 0
    total_time_taken = 0

    for predicted_answers, maj1_at_16, avg_time in results:
        all_predicted_answers.extend(predicted_answers)
        total_maj1_at_16_count += maj1_at_16 * len(predicted_answers)
        total_time_taken += avg_time * len(predicted_answers)

    # Calculate overall averages
    total_questions = len(dataset)
    avg_maj1_at_16_score = total_maj1_at_16_count / total_questions
    avg_time_per_problem = total_time_taken / total_questions

    # Save aggregated results to a file
    output_df = pd.DataFrame(all_predicted_answers)
    output_df.to_csv('predicted_answers_for_maj1_16_mistrali_Mathstral_7B.csv', index=False)

    return avg_maj1_at_16_score, avg_time_per_problem

# Start total evaluation time tracking
start_total_time = time.time()

# Load dataset
dataset = pd.read_json('scoredataset.json')

# Run main evaluation
avg_maj1_at_16_score, avg_time_per_problem = main_evaluation(dataset)

# End total evaluation time tracking
end_total_time = time.time()
total_elapsed_time = end_total_time - start_total_time

print(f"Maj1@16 Score: {avg_maj1_at_16_score:.2f}%")
print(f"Average Time Taken per Problem: {avg_time_per_problem:.2f} seconds")
print(f"Total Time Taken for Model Run and Evaluation: {total_elapsed_time:.2f} seconds")


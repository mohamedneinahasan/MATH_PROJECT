import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import time
import multiprocessing

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Split the dataset for two GPUs
def split_dataset(dataset, num_splits=2):
    split_size = len(dataset) // num_splits
    return [dataset.iloc[i*split_size: (i+1)*split_size] for i in range(num_splits)]

# Function to solve math problems on a specific GPU
def solve_math_problem(model, problem_text, device):
    prompt = f"Solve the following math problem: {problem_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    
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
    return output_text

# Evaluation function for ROUGE and BLEU on a specific GPU
def evaluate_on_gpu(gpu_id, dataset_split):
    # Set device
    device = torch.device(f"cuda:{gpu_id}")
    
    # Load model on the specific GPU
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    rouge_scorer_instance = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    bleu_scores = []
    predicted_answers_list = []
    total_time_taken = 0

    for idx, row in dataset_split.iterrows():
        print(f"[GPU {gpu_id}] Solving problem ID {idx + 1}/{len(dataset_split)}...")

        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Track the start time for solving the problem
        start_time = time.time()

        # Generate a predicted answer
        predicted_answer = solve_math_problem(model, problem_text, device)

        # Track the end time after solving the problem
        end_time = time.time()
        time_taken = end_time - start_time
        total_time_taken += time_taken

        # Calculate ROUGE-L and BLEU scores
        rouge_score = rouge_scorer_instance.score(reference_answer, predicted_answer)['rougeL'].fmeasure
        rouge_scores.append(rouge_score)

        reference_tokens = reference_answer.split()
        predicted_tokens = predicted_answer.split()
        bleu_score = sentence_bleu([reference_tokens], predicted_tokens)
        bleu_scores.append(bleu_score)

        # Store results for analysis
        row['predicted_answer'] = predicted_answer
        row['rougeL_score'] = rouge_score
        row['bleu_score'] = bleu_score
        predicted_answers_list.append(row)

    # Calculate average ROUGE and BLEU scores
    avg_rouge_score = sum(rouge_scores) / len(rouge_scores)
    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    avg_time_taken = total_time_taken / len(dataset_split)

    return predicted_answers_list, avg_rouge_score, avg_bleu_score, avg_time_taken

# Run evaluation on two GPUs and aggregate results
def main_evaluation(dataset):
    # Split dataset for each GPU
    dataset_splits = split_dataset(dataset, num_splits=2)

    # Start parallel processing on two GPUs
    with multiprocessing.Pool(2) as pool:
        results = pool.starmap(evaluate_on_gpu, [(0, dataset_splits[0]), (1, dataset_splits[1])])

    # Aggregate results from both GPUs
    all_predicted_answers = []
    total_rouge_score = total_bleu_score = total_time_taken = 0

    for predicted_answers, avg_rouge, avg_bleu, avg_time in results:
        all_predicted_answers.extend(predicted_answers)
        total_rouge_score += avg_rouge * len(predicted_answers)
        total_bleu_score += avg_bleu * len(predicted_answers)
        total_time_taken += avg_time * len(predicted_answers)

    # Calculate overall averages
    total_questions = len(dataset)
    avg_rouge_score = total_rouge_score / total_questions
    avg_bleu_score = total_bleu_score / total_questions
    avg_time_per_problem = total_time_taken / total_questions

    # Save aggregated results to a file
    output_df = pd.DataFrame(all_predicted_answers)
    output_df.to_csv('predicted_answers_BELU_ROUGE_FOR_meta__Llama_3.1_8B.csv', index=False)

    return avg_rouge_score, avg_bleu_score, avg_time_per_problem

# Start total evaluation time tracking
start_total_time = time.time()

# Load dataset
dataset = pd.read_json('scoredataset.json')

# Run main evaluation
avg_rouge_score, avg_bleu_score, avg_time_per_problem = main_evaluation(dataset)

# End total evaluation time tracking
end_total_time = time.time()
total_elapsed_time = end_total_time - start_total_time

print(f"Average ROUGE-L Score: {avg_rouge_score:.4f}")
print(f"Average BLEU Score: {avg_bleu_score:.4f}")
print(f"Average Time Taken per Problem: {avg_time_per_problem:.2f} seconds")
print(f"Total Time Taken for Model Run and Evaluation: {total_elapsed_time:.2f} seconds")


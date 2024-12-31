import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from fuzzywuzzy import fuzz
import time  # Import time module for tracking time

# Start total evaluation time tracking
start_total_time = time.time()

# Load the model and tokenizer
model_name = "meta-math/MetaMath-Mistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to solve math problems with multiple passes (for Maj1@64)
def solve_math_problem(problem_text, num_attempts=1):
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

# Load dataset
dataset = pd.read_json('100scoredataset.json')

# Evaluation function for Maj1@64, RR.All, RR.Top-8, and fuzzy score
def evaluate_model(dataset):
    total_questions = len(dataset)
    maj1 = 0  # Counter for Maj1@64
    rr_all = 0  # Counter for RR.All
    rr_top_8 = 0  # Counter for RR.Top-8
    fuzzy_scores = []
    predicted_answers_list = []
    total_time_taken = 0  # Track total time taken for all problems

    for idx, row in dataset.iterrows():
        print(f"Solving problem ID {idx + 1}/{total_questions}...")

        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Track the start time for solving the problem
        start_time = time.time()

        # Get predictions for multiple attempts
        predicted_answers = solve_math_problem(problem_text, num_attempts=64)

        # Track the end time after solving the problem
        end_time = time.time()
        time_taken = end_time - start_time
        total_time_taken += time_taken

        # Calculate Maj1@64: check if at least 64% of predictions are correct
        correct_answers_count = sum(exact_match(answer, reference_answer) for answer in predicted_answers)
        if correct_answers_count >= 41:  # 64% of 64 attempts
            maj1 += 1

        # Calculate RR.All: check if the correct answer appears in any of the generated answers
        if any(exact_match(answer, reference_answer) for answer in predicted_answers):
            rr_all += 1

        # Calculate RR.Top-8: check if the correct answer is within the top 8 generated answers
        if any(exact_match(answer, reference_answer) for answer in predicted_answers[:8]):
            rr_top_8 += 1

        # Calculate fuzzy score for the first predicted answer
        fuzzy = fuzzy_score(predicted_answers[0], reference_answer)
        fuzzy_scores.append(fuzzy)

        # Store predicted answer for later analysis
        row['predicted_answer'] = predicted_answers[0]
        row['fuzzy_score'] = fuzzy
        predicted_answers_list.append(row)

        # Log time taken for each problem
        print(f"Time taken for problem ID {idx + 1}: {time_taken:.2f} seconds")

    # Calculate Maj1@64 rate, RR.All rate, RR.Top-8 rate, and Average Fuzzy Score
    maj1_rate = maj1 / total_questions * 100
    rr_all_rate = rr_all / total_questions * 100
    rr_top_8_rate = rr_top_8 / total_questions * 100
    avg_fuzzy_score = sum(fuzzy_scores) / total_questions
    avg_time_taken = total_time_taken / total_questions  # Average time per problem

    # Save results to a file
    output_df = pd.DataFrame(predicted_answers_list)
    output_df.to_csv('predicted_answers_for_MetaMath_Mistral_7B', index=False)

    return maj1_rate, rr_all_rate, rr_top_8_rate, avg_fuzzy_score, avg_time_taken

# Run the evaluation on your dataset and display results
maj1_rate, rr_all_rate, rr_top_8_rate, avg_fuzzy_score, avg_time_taken = evaluate_model(dataset)

# End total evaluation time tracking
end_total_time = time.time()
total_elapsed_time = end_total_time - start_total_time

print(f"Maj1@64 Accuracy: {maj1_rate:.2f}%")
print(f"RR.All Accuracy: {rr_all_rate:.2f}%")
print(f"RR.Top-8 Accuracy: {rr_top_8_rate:.2f}%")
print(f"Average Fuzzy Matching Score: {avg_fuzzy_score:.2f}")
print(f"Average Time Taken per Problem: {avg_time_taken:.2f} seconds")
print(f"Total Time Taken for Model Run and Evaluation: {total_elapsed_time:.2f} seconds")


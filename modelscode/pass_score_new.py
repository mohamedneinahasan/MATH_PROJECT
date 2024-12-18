import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from fuzzywuzzy import fuzz

# Load the model and tokenizer
model_name = "mistralai/Mathstral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to solve math problems with multiple passes (for pass@10)
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
dataset = pd.read_json('scoredataset.json')

# Evaluation function for pass@1, pass@2, pass@10, and fuzzy score
def evaluate_model(dataset):
    total_questions = len(dataset)
    pass1 = pass2 = pass10 = 0
    fuzzy_scores = []
    predicted_answers_list = []

    for idx, row in dataset.iterrows():
        print(f"Solving problem ID {idx + 1}/{total_questions}...")

        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Get predictions for multiple attempts
        predicted_answers = solve_math_problem(problem_text, num_attempts=10)

        # Calculate pass@1, pass@2, pass@10
        pass1 += exact_match(predicted_answers[0], reference_answer)
        pass2 += any(exact_match(answer, reference_answer) for answer in predicted_answers[:2])
        pass10 += any(exact_match(answer, reference_answer) for answer in predicted_answers)

        # Calculate fuzzy score for the first predicted answer
        fuzzy = fuzzy_score(predicted_answers[0], reference_answer)
        fuzzy_scores.append(fuzzy)

        # Store predicted answer for later analysis
        row['predicted_answer'] = predicted_answers[0]
        row['fuzzy_score'] = fuzzy
        predicted_answers_list.append(row)

    # Calculate Pass@1, Pass@2, Pass@10, and Average Fuzzy Score
    pass1_rate = pass1 / total_questions * 100
    pass2_rate = pass2 / total_questions * 100
    pass10_rate = pass10 / total_questions * 100
    avg_fuzzy_score = sum(fuzzy_scores) / total_questions

    # Save results to a file
    output_df = pd.DataFrame(predicted_answers_list)
    output_df.to_csv('predicted_answers_with_mathstral_7b_0.1.csv', index=False)

    return pass1_rate, pass2_rate, pass10_rate, avg_fuzzy_score

# Run the evaluation on your dataset and display results
pass1_rate, pass2_rate, pass10_rate, avg_fuzzy_score = evaluate_model(dataset)
print(f"Pass@1 Accuracy: {pass1_rate:.2f}%")
print(f"Pass@2 Accuracy: {pass2_rate:.2f}%")
print(f"Pass@10 Accuracy: {pass10_rate:.2f}%")
print(f"Average Fuzzy Matching Score: {avg_fuzzy_score:.2f}")


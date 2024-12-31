import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from fuzzywuzzy import fuzz

# Load the model and tokenizer
model_name = "THUDM/glm-4-9b"
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
    # Explicitly handle padding without using `padding=True`
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # If tokenizer doesn't have padding, handle manually
    if tokenizer.pad_token is not None:
        inputs = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512)
    else:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    inputs = inputs.to(device)

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

# Evaluation function for Maj1@64 and fuzzy score
def evaluate_model(dataset):
    total_questions = len(dataset)
    maj1 = 0  # Counter for Maj1@64
    fuzzy_scores = []
    predicted_answers_list = []

    for idx, row in dataset.iterrows():
        print(f"Solving problem ID {idx + 1}/{total_questions}...")

        problem_text = row['algebraic_word_problem']
        reference_answer = row['reference_answer']

        # Get predictions for multiple attempts
        predicted_answers = solve_math_problem(problem_text, num_attempts=64)

        # Calculate Maj1@64: check if at least 64% of predictions are correct
        correct_answers_count = sum(exact_match(answer, reference_answer) for answer in predicted_answers)
        if correct_answers_count >= 41:  # 64% of 64 attempts
            maj1 += 1

        # Calculate fuzzy score for the first predicted answer
        fuzzy = fuzzy_score(predicted_answers[0], reference_answer)
        fuzzy_scores.append(fuzzy)

        # Store predicted answer for later analysis
        row['predicted_answer'] = predicted_answers[0]
        row['fuzzy_score'] = fuzzy
        predicted_answers_list.append(row)

    # Calculate Maj1@64 rate and Average Fuzzy Score
    maj1_rate = maj1 / total_questions * 100
    avg_fuzzy_score = sum(fuzzy_scores) / total_questions

    # Save results to a file
    output_df = pd.DataFrame(predicted_answers_list)
    output_df.to_csv('predicted_answers_for_glm_4_9b.csv', index=False)

    return maj1_rate, avg_fuzzy_score

# Run the evaluation on your dataset and display results
maj1_rate, avg_fuzzy_score = evaluate_model(dataset)
print(f"Maj1@64 Accuracy: {maj1_rate:.2f}%")
print(f"Average Fuzzy Matching Score: {avg_fuzzy_score:.2f}")


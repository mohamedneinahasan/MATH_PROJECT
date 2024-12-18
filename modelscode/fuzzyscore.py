import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from fuzzywuzzy import fuzz

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the padding token to EOS token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to solve math problems with a single pass
def solve_math_problem(problem_text):
    prompt = f"Solve the following math problem: {problem_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],  # Set the attention mask
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id  # Use EOS token as padding token
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return output_text

# Function to compute Exact Match and Accuracy
def evaluate_prediction(predicted_answer, reference_answer):
    # Exact Match (EM)
    em = int(predicted_answer.strip() == reference_answer.strip())

    # Fuzzy Matching for flexible comparison
    fuzzy_score = fuzz.ratio(predicted_answer.strip(), reference_answer.strip())

    return em, fuzzy_score

# Load dataset
dataset = pd.read_json('100scoredataset.json')

# Evaluation function to loop through the dataset with pass@1 metric
def evaluate_model(dataset):
    total_questions = len(dataset)
    pass1 = 0
    fuzzy_scores = []

    for _, row in dataset.iterrows():
        predicted_answer = solve_math_problem(row['algebraic_word_problem'])
        reference_answer = row['reference_answer']

        # Calculate Exact Match and Fuzzy score for pass@1
        em, fuzzy_score = evaluate_prediction(predicted_answer, reference_answer)

        # Track pass@1 results
        pass1 += em
        fuzzy_scores.append(fuzzy_score)

        # Store the predicted answer in dataset for later analysis
        row['predicted_answer'] = predicted_answer

    # Calculate Pass@1 rate and Average Fuzzy Score
    pass1_rate = pass1 / total_questions * 100
    avg_fuzzy_score = sum(fuzzy_scores) / total_questions

    return pass1_rate, avg_fuzzy_score

# Run the evaluation on your dataset and display results
pass1_rate, avg_fuzzy_score = evaluate_model(dataset)
print(f"Pass@1 Accuracy: {pass1_rate:.2f}%")
print(f"Average Fuzzy Matching Score: {avg_fuzzy_score:.2f}")


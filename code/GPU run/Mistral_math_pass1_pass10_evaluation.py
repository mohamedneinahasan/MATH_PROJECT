import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd

# Load the model and tokenizer
model_name = "fdqerq22ds/MathScale-Mistral"
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
                max_new_tokens=50,  # Adjusted for efficiency
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        outputs.append(output_text)
    return outputs

# Function to compute Exact Match
def exact_match(predicted_answer, reference_answer):
    return int(predicted_answer.strip() == reference_answer.strip())

# Load dataset
dataset = pd.read_json('/content/1000scoredataset.json')

# Evaluation function for pass@1 and pass@10 with batching
def evaluate_model(dataset, batch_size=100):
    total_questions = len(dataset)
    pass1 = 0
    pass10 = 0
    predictions = []

    for start in range(0, total_questions, batch_size):
        batch = dataset.iloc[start:start+batch_size]

        for _, row in batch.iterrows():
            predicted_answers = solve_math_problem(row['algebraic_word_problem'], num_attempts=10)
            reference_answer = row['reference_answer']

            # Calculate pass@1 and pass@10
            pass1 += exact_match(predicted_answers[0], reference_answer)
            pass10 += any(exact_match(answer, reference_answer) for answer in predicted_answers)

            # Store the first predicted answer for each row
            row['predicted_answer'] = predicted_answers[0]
            predictions.append(row)

        # Save after each batch
        result_df = pd.DataFrame(predictions)
        result_df.to_json(f'/content/scored_dataset_with_predictions_batch_{start}.json', orient='records', lines=True)
        result_df.to_csv(f'/content/scored_dataset_with_predictions_batch_{start}.csv', index=False)

    # Calculate Pass@1 and Pass@10 rates
    pass1_rate = pass1 / total_questions * 100
    pass10_rate = pass10 / total_questions * 100

    return pass1_rate, pass10_rate

# Run the evaluation on your dataset and display results
pass1_rate, pass10_rate = evaluate_model(dataset)
print(f"Pass@1 Accuracy: {pass1_rate:.2f}%")
print(f"Pass@10 Accuracy: {pass10_rate:.2f}%")

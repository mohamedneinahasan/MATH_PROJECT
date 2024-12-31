import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import pandas as pd
from fuzzywuzzy import fuzz

# Load the model and tokenizer
model_name = "DeepMount00/Mistral-RAG"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Function to solve math problems
def solve_math_problem(problem_text):
    prompt = f"Solve the following math problem: {problem_text}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text.strip()

# Function to compute Exact Match and Accuracy
def evaluate_prediction(predicted_answer, reference_answer):
    # Exact Match (EM)
    em = int(predicted_answer.strip() == reference_answer.strip())

    # Fuzzy Matching for flexible comparison
    fuzzy_score = fuzz.ratio(predicted_answer.strip(), reference_answer.strip())

    return em, fuzzy_score

# Load dataset
dataset = pd.read_json('/Users/mohamedneinahasan/Downloads/project/datasets /scoredataset.json')

# Evaluation function to loop through the dataset
def evaluate_model(dataset):
    total_questions = len(dataset)
    total_correct = 0
    fuzzy_scores = []

    for _, row in dataset.iterrows():
        predicted_answer = solve_math_problem(row['algebraic_word_problem'])
        reference_answer = row['reference_answer']
        
        # Get Exact Match and Fuzzy score
        em, fuzzy_score = evaluate_prediction(predicted_answer, reference_answer)
        
        total_correct += em
        fuzzy_scores.append(fuzzy_score)
        
        # Store predicted answer in dataset for later analysis
        row['predicted_answer'] = predicted_answer

    # Calculate Accuracy and Average Fuzzy Score
    accuracy = total_correct / total_questions * 100
    avg_fuzzy_score = sum(fuzzy_scores) / total_questions

    return accuracy, avg_fuzzy_score

# Gradio interface for individual problem-solving
def gradio_interface(problem_text):
    solution = solve_math_problem(problem_text)
    return f"Problem: {problem_text}\n\nSolution:\n{solution}"

# Launch Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs="text",
    outputs="text",
    title="Math Problem Solver with Mistral-RAG",
    description="Enter a math problem and the AI will solve it step by step.",
    examples=[
        ["John has 5 apples and buys 3 more each day. How many apples after 4 days?"],
        ["If you invest $2000 at a rate of 5% simple interest per year, how much interest will you earn in 3 years?"]
    ]
)

# Run the evaluation on your dataset and display results
accuracy, avg_fuzzy_score = evaluate_model(dataset)
print(f"Accuracy: {accuracy:.2f}%")
print(f"Average Fuzzy Matching Score: {avg_fuzzy_score:.2f}")

iface.launch()

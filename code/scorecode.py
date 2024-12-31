!pip install transformers gradio torch pandas rouge-score datasets nltk fuzzywuzzy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import pandas as pd
import json
from rouge_score import rouge_scorer
from datasets import load_metric
from fuzzywuzzy import fuzz
import nltk

# Load BLEU metric and other resources
nltk.download('punkt')
bleu_metric = load_metric("bleu")

# Load ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Load the model and tokenizer
model_name = "DeepMount00/Mistral-RAG"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load your dataset (adjusted for your dataset structure)
with open('scoredataset.json', 'r') as f:
    dataset = json.load(f)

# Convert dataset into a DataFrame for easier access
df = pd.DataFrame(dataset)

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
    return output_text

# Function to calculate metrics (ROUGE, BLEU, Fuzzy)
def calculate_metrics(reference_answer, predicted_answer):
    # ROUGE score
    rouge_scores = scorer.score(reference_answer, predicted_answer)
    
    # BLEU score
    reference_tokens = [nltk.word_tokenize(reference_answer.lower())]
    predicted_tokens = nltk.word_tokenize(predicted_answer.lower())
    bleu_score = bleu_metric.compute(predictions=[predicted_tokens], references=[reference_tokens])

    # Fuzzy matching score
    fuzzy_score = fuzz.ratio(reference_answer, predicted_answer)

    return rouge_scores, bleu_score['bleu'], fuzzy_score

# Gradio interface
def gradio_interface(problem_text, reference_answer):
    # Generate predicted answer
    predicted_answer = solve_math_problem(problem_text)

    # Calculate metrics
    rouge_scores, bleu_score, fuzzy_score = calculate_metrics(reference_answer, predicted_answer)

    # Format the output
    output = (
        f"Problem: {problem_text}\n\n"
        f"Predicted Answer:\n{predicted_answer}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Evaluation Metrics:\n"
        f"ROUGE-1: {rouge_scores['rouge1'].fmeasure:.4f}\n"
        f"ROUGE-2: {rouge_scores['rouge2'].fmeasure:.4f}\n"
        f"ROUGE-L: {rouge_scores['rougeL'].fmeasure:.4f}\n"
        f"BLEU: {bleu_score:.4f}\n"
        f"Fuzzy Matching: {fuzzy_score:.2f}%\n"
    )

    return output

# Launch Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=["text", "text"],  # Takes the problem and reference answer
    outputs="text",
    title="Math Problem Solver with Evaluation Metrics",
    description="Enter a math problem and reference answer. The AI will solve it and evaluate the solution using ROUGE, BLEU, and Fuzzy Matching.",
    examples=[[row['algebraic_word_problem'], row['reference_answer']] for idx, row in df.iterrows()]  # Load examples from your dataset
)

iface.launch()

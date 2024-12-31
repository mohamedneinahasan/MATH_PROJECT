!pip install transformers gradio torch pandas

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import pandas as pd

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
    return output_text

# Gradio interface
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
        ["If John has 5 apples and buys 3 more every day, how many apples will he have after 4 days?"],
        ["If you invest $2000 at a rate of 5% simple interest per year, how much interest will you earn in 3 years?"]
    ]
)

iface.launch()

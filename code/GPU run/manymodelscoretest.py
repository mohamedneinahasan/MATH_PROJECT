import gradio as gr
from sklearn.metrics import accuracy_score, f1_score
from fuzzywuzzy import fuzz
import numpy as np

# Load models - replace with actual model loading
models = {
    "Model 1": load_model_1(),
    "Model 2": load_model_2(),
    "Model 3": load_model_3(),
    "Model 4": load_model_4(),
    "Model 5": load_model_5(),
}

# Example datasets
datasets = {
    "Your Dataset": load_custom_dataset(),
    "Dataset 2": load_dataset_2(),
    "Dataset 3": load_dataset_3(),
}

# Evaluation Metrics
def exact_match(pred, truth):
    return int(pred == truth)

def fuzzy_match(pred, truth):
    return fuzz.ratio(pred, truth)

# Inference with each model
def generate_solution(model_name, question):
    model = models[model_name]
    # Assume generate_answer_with_steps returns a list of steps for solution
    steps = model.generate_answer_with_steps(question)  
    final_answer = steps[-1]  # Last step as the final answer
    return steps, final_answer

# Function to evaluate models on datasets
def evaluate_model_on_dataset(model_name, dataset_name):
    model = models[model_name]
    dataset = datasets[dataset_name]
    
    exact_matches = []
    fuzzy_scores = []
    
    for question, true_answer in dataset:
        _, prediction = generate_solution(model_name, question)
        exact_matches.append(exact_match(prediction, true_answer))
        fuzzy_scores.append(fuzzy_match(prediction, true_answer))
    
    accuracy = np.mean(exact_matches)
    avg_fuzzy = np.mean(fuzzy_scores)
    
    return accuracy, avg_fuzzy

# Gradio Interface Function
def gradio_interface(question, model_name, dataset_name):
    # Generate step-by-step answer
    steps, answer = generate_solution(model_name, question)
    
    # Evaluate on the selected dataset
    accuracy, fuzzy_score = evaluate_model_on_dataset(model_name, dataset_name)
    
    # Display results
    response = {
        "Steps": steps,
        "Answer": answer,
        "Accuracy": accuracy,
        "Fuzzy Match": fuzzy_score
    }
    return response

# Setting up the Gradio Interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.inputs.Textbox(label="Question"),
        gr.inputs.Dropdown(list(models.keys()), label="Model"),
        gr.inputs.Dropdown(list(datasets.keys()), label="Dataset")
    ],
    outputs=[
        gr.outputs.JSON(label="Steps and Final Answer"),
        gr.outputs.Textbox(label="Accuracy"),
        gr.outputs.Textbox(label="Fuzzy Match Score")
    ],
    live=True
)

iface.launch()

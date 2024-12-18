import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the dataset
dataset = load_dataset("json", data_files={"data": "scoredataset.json"})["data"]

# Step 2: Split the dataset into training and testing sets using Hugging Face's split
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Step 3: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 4: Load the model and adjust token embeddings if necessary
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model.resize_token_embeddings(len(tokenizer))

# Step 5: Preprocessing function
def preprocess_function(examples, tokenizer, max_length=512):
    inputs = examples["algebraic_word_problem"]  # Column for questions
    targets = examples["step_by_step_solution"]  # Column for step-by-step solutions
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 6: Tokenize the dataset
tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
tokenized_test = test_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save model checkpoints
    eval_strategy="epoch",    # Evaluate at the end of each epoch
    learning_rate=5e-5,             # Learning rate
    per_device_train_batch_size=8,  # Training batch size per GPU
    per_device_eval_batch_size=8,   # Evaluation batch size per GPU
    num_train_epochs=3,             # Number of epochs
    weight_decay=0.01,              # Weight decay
    save_total_limit=2,             # Limit the number of saved checkpoints
    save_strategy="epoch",          # Save at the end of each epoch
    fp16=True,                      # Use mixed precision for faster training
    logging_dir="./logs",           # Directory for logs
    logging_steps=10,
    gradient_accumulation_steps=2,  # Gradient accumulation for large batches
    report_to="none",               # Disable integration with logging tools like WandB
    load_best_model_at_end=True,    # Load the best model at the end of training
)

# Step 8: Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()}
)

# Step 9: Fine-tune the model
trainer.train()

# Step 10: Save the fine-tuned model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 11: Evaluate the model on the test set
metrics = trainer.evaluate()
print("Evaluation Metrics:", metrics)

print("Fine-tuning and evaluation completed!")

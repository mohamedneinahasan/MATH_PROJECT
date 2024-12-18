import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Load the dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Step 2: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# Step 3: Ensure the tokenizer has a padding token
if tokenizer.pad_token is None:
    # Use eos_token as pad_token if available
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        # Add a new [PAD] token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 4: Load the model and resize token embeddings if necessary
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
model.resize_token_embeddings(len(tokenizer))

# Step 5: Preprocessing function
def preprocess_function(examples, tokenizer, max_length=512):
    inputs = examples["John takes 4 hours to paint a wall, while Mary takes 6 hours. How long will it take them to paint the wall together?"]  # Replace "problem" with the column name for math problems in your dataset
    targets = examples["2.4 hours"]  # Replace "solution" with the column name for solutions
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Step 6: Tokenize the dataset
tokenized_datasets = dataset.map(
    lambda x: preprocess_function(x, tokenizer),
    batched=True
)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",          # Directory to save model checkpoints
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
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
    ddp_find_unused_parameters=False,  # Optimize for multi-GPU
)

# Step 8: Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
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

# Additional Notes:
# - Replace `train.csv` and `test.csv` with the actual paths to your dataset.
# - Make sure the dataset columns are named `problem` and `solution` or adjust the preprocessing function accordingly.
# - If you are running on multiple GPUs, `torch.distributed.launch` will automatically handle parallelization.

print("Fine-tuning and evaluation completed!")

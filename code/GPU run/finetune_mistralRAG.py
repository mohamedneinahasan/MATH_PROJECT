from datasets import load_dataset, DatasetDict

# Load dataset and split it into train and test sets
dataset = load_dataset("json", data_files="scoredataset.json")
dataset = dataset["train"].train_test_split(test_size=0.2)  # 80% train, 20% test

# Load model and tokenizer
model_name = "DeepMount00/Mistral-RAG"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize function
def tokenize_function(example):
    inputs = tokenizer(example['algebraic_word_problem'], padding="max_length", truncation=True)
    labels = tokenizer(example['reference_answer'], padding="max_length", truncation=True)
    
    inputs["labels"] = labels["input_ids"]
    return inputs

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
    remove_unused_columns=False
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)

# Train model
trainer.train()

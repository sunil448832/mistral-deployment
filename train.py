from transformers import AutoTokenizer
import torch
import datasets
import numpy as np
from functools import partial
from dataset import VicunaDataset,collate_fn

from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments,Trainer
import torch
from transformers import BitsAndBytesConfig
from utils import print_number_of_trainable_model_parameters



from transformers import AutoTokenizer
data_url='https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
model_name="mistralai/Mistral-7B-Instruct-v0.1"
max_length=512
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

print("Preparing dataset...")
dataset=VicunaDataset(data_url,tokenizer,max_length)()
print("Done!")
collate_fn=partial(collate_fn,tokeniser=tokenizer)



lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=16,
    target_modules=["k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model_name="mistralai/Mistral-7B-Instruct-v0.1"
nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            )

original_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=nf4_config)

peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

import time
output_dir = f'outputs-{str(int(time.time()))}'

# Define a function to save the model when validation loss decreases
def save_checkpoint(trainer):
    if trainer.state.is_world_process_zero:
        checkpoint = trainer.state.best_model_checkpoint
        if checkpoint is not None:
            trainer.save_model(checkpoint)

# Define a function to evaluate the model at every 100 steps
def evaluate(eval_steps):
    def _callback_function(state):
        if state.global_step % eval_steps == 0 and state.global_step > 0:
            metrics = state.trainer.evaluate()
            state.log(metrics)

    return _callback_function

# Create TrainingArguments with additional configurations
peft_training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    eval_steps=20,
    learning_rate=1e-4,
    save_total_limit=1,  # Save only the best checkpoint
    load_best_model_at_end=True,  # Load the best checkpoint at the end of training
    evaluation_strategy="steps",  # Evaluate at specific steps
    save_strategy="steps",  # Save model at specific steps
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=20,  # Log at every 10 steps
    save_steps=20,  # Save model at every 10 steps
)

trainer = Trainer(
    # Replace these with your model, tokenizer, and data
    model=peft_model,
    args=peft_training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['val'],
    tokenizer=tokenizer,
    callbacks=[
        save_checkpoint,  # Save model based on validation loss
        evaluate(eval_steps=100),  # Evaluate at every 100 steps
    ]
)

trainer.train()
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
from transformers import  get_cosine_schedule_with_warmup
from accelerate import Accelerator



from transformers import AutoTokenizer
accelerator = Accelerator()

data_url='https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
model_name="mistralai/Mistral-7B-Instruct-v0.1"
max_length=512
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

print("Preparing dataset...")
dataset=VicunaDataset(data_url,tokenizer,max_length=max_length,num_samples=10000)()
print("Done!")
collate_fn=partial(collate_fn,tokeniser=tokenizer)



lora_config = LoraConfig(
    r=16, # Rank
    lora_alpha=32,
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
print("Downloading Model...")
original_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=nf4_config)

print("Done!")
peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

peft_model, train_loader, eval_loader = accelerator.prepare(
    peft_model, dataset['train'], dataset['test']
)

import time
output_dir = 'outputs'

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
    overwrite_output_dir = 'True',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=6,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    eval_steps=200,
    learning_rate=1e-4,
    lr_scheduler_type='cosine',
    save_total_limit=1,  # Save only the best checkpoint
    load_best_model_at_end=True,  # Load the best checkpoint at the end of training
    evaluation_strategy="steps",  # Evaluate at specific steps
    save_strategy="steps",  # Save model at specific steps
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=200,  # Log at every 10 steps
    save_steps=200,  # Save model at every 10 steps
)

trainer = Trainer(
    # Replace these with your model, tokenizer, and data
    model=peft_model,
    args=peft_training_args,
    train_dataset=train_loader,
    eval_dataset=eval_loader,
    tokenizer=tokenizer,
    data_collator=collate_fn,
)

trainer.train()
if accelerator.is_main_process:
    trainer.save_model(output_dir='weights')



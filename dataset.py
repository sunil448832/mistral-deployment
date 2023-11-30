import sys
import os
from typing import Any
import torch
from datasets import Dataset,load_dataset
from tqdm import tqdm

import requests

INST_START_TEMPLATE = "<s>[INST] "
INST_END_TEMPLATE = " [/INST]\n"
ANS_END_TEMPLATE = " </s>\n"
IGNORE_INDEX = -100

def create_prompt(dialog):
    sys_msg = 'You are a smart chatbot. Provide answers correctly'
    conversation = f"{INST_START_TEMPLATE}{sys_msg}\n\n"
    prev_from_human = False

    for idx, sentence in enumerate(dialog):
        sentence_from = sentence["from"].lower()
        if (sentence_from == "human") and (not prev_from_human) and (idx != len(dialog) - 1):
            if idx == 0:
                conversation += sentence["value"] + INST_END_TEMPLATE
            else:
                conversation += INST_START_TEMPLATE + sentence["value"] + INST_END_TEMPLATE
            prev_from_human = True
        elif prev_from_human:
            conversation += sentence["value"] + ANS_END_TEMPLATE
            prev_from_human = False

    return conversation


def preprocess(prompt, tokenizer, max_length=1024):
    q_list, a_list = [], []
    question = ''
    ans_end_template_token_ids = tokenizer(ANS_END_TEMPLATE)

    for sentence in prompt.split(ANS_END_TEMPLATE)[:-1]:
        q, a = sentence.split(INST_END_TEMPLATE)
        question = question + q + INST_END_TEMPLATE
        tokenised_question = tokenizer(question)

        ans_max_len = max_length - len(tokenised_question['input_ids']) - len(ans_end_template_token_ids['input_ids'])
        if ans_max_len < 1:
            continue
        tokenised_answer = tokenizer(a, max_length=ans_max_len, truncation=True)
        tokenised_answer['input_ids'] = [IGNORE_INDEX] * len(tokenised_question['input_ids']) + \
                                        tokenised_answer['input_ids'] + \
                                        ans_end_template_token_ids['input_ids']

        tokenised_answer['attention_mask'] = [1] * len(tokenised_answer['input_ids'])
        q_list.append(tokenised_question)
        a_list.append(tokenised_answer)

    return q_list, a_list




def build_dataset(dataset,tokenizer,max_seq_len):
  dataset=dataset.map(lambda example:{"prompt": create_prompt(example["conversations"])})
  inputs,labels=[],[]
  for example in tqdm(dataset):
    prompt=example['prompt']
    q_prompt_tokenised,a_prompt_tokenized=preprocess(prompt,tokenizer,max_seq_len)
    inputs.extend(q_prompt_tokenised)
    labels.extend(a_prompt_tokenized)


  data_dict={"inputs":inputs,
        "labels":labels}
  dataset = Dataset.from_dict(data_dict)

  dataset = dataset.train_test_split(test_size=0.2, shuffle=False, seed=42)
  return dataset

def collate_fn(batch,tokeniser):
    max_len_input=max([len(x['input_ids']) for x in batch])
    max_len_label=max([len(x['labels']) for x in batch])
    max_len=max(max_len_input,max_len_label)
    data={'input_ids':[] ,'attention_mask':[],'labels':[]}

    for sample in batch:
      data['input_ids'].append(sample['input_ids']+[tokeniser.pad_token_id]*(max_len-len(sample['input_ids'])))
      data['attention_mask'].append(sample['attention_mask']+[0]*(max_len-len(sample['attention_mask'])))
      data['labels'].append(sample['labels']+[IGNORE_INDEX]*(max_len-len(sample['labels'])))

    batch={k:torch.tensor(v) for k,v in data.items()}
    return batch

def download(url,file_name):
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(file_name, 'wb') as file, tqdm(
                desc=file_name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                progress_bar.update(len(data))

    except Exception as e:
        print("An error occurred:", str(e))

class VicunaDataset:
    def __init__(self,download_url,tokenizer,max_length=1024) -> None:
        os.makedirs('data', exist_ok=True)
        self.data_path=os.path.join('data','sharegpt.json')
        if not os.path.exists( self.data_path):
           print("Downloading data ...")
           download(download_url, self.data_path)
           print(f"File downloaded and saved as {self.data_path}")

        else:
            print(f"Data already exists at { self.data_path}. Skipping download.")
        self.tokenizer=tokenizer
        self.max_length=max_length
    
    def __call__(self):
        dataset=load_dataset("json", data_files= self.data_path)
        dataset=dataset['train'].select(range(10000))
        dataset=build_dataset(dataset,self.tokenizer,self.max_length)
        return dataset

if __name__=='__main__':
    from transformers import AutoTokenizer
    data_url='https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
    model_name="mistralai/Mistral-7B-Instruct-v0.1"
    max_length=512
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Preparing dataset...")
    dataset=VicunaDataset(data_url,tokenizer,max_length)()
    print("Done!")
    
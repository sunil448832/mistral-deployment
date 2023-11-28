from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import BitsAndBytesConfig

# Define a Language Model class
class MistralModel:
    def __init__(self, model_name):
        
        # Load the pre-trained language model with specific settings

        model_name="mistralai/Mistral-7B-Instruct-v0.1"
        nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=nf4_config)

        
        # Initialize the tokenizer for the same model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Set custom padding token and padding side
        self.tokenizer.pad_token = "[PAD]"
        self.tokenizer.padding_side = "left"

    def generate_response(self, messages, max_tokens=100, do_sample=True):
        # Tokenize the input messages and move them to the selected device (GPU or CPU)
        input_ids = self.tokenizer(
            messages,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).input_ids.cuda()
        
        with torch.no_grad():
            # Generate a response using the loaded model
            generated_ids = self.model.generate(
                input_ids,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=0.3  # Adjust the sampling temperature
            )
            # Decode the generated tokens into a human-readable response
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        return response



class MistralPrompts:
    
    # Create a standalone question prompt by using chat history and followup question.
    @staticmethod
    def create_standalone_question_prompt(question, chat_history):
      instruction='aking chat history as context, rephrase follow up question into a standalone question'
      return MistralPrompts.create_question_prompt(question, chat_history)

    # Create a chat history prompt by adding context and question to a chat history prompt.
    @staticmethod
    def create_prompt(instruction,chat_history, question):
        if len(chat_history) == 0:
            return f"<s>[INST]{instruction}\n\n Question: {question}[/INST]"
        else:
          user_message, bot_message = chat_history[0]
          chat_history_text = f"<s>[INST]{instruction}\n\n{user_message} [/INST] {bot_message}</s>"
          chat_history_text += "".join(f"[INST] {user_message} [/INST] {bot_message}</s>" for user_message, bot_message in chat_history[1:])
          return chat_history_text+f"[INST] Question: {question}[/INST]"

     

    # Create a question prompt by adding context and question to a chat history prompt.
    @staticmethod
    def create_question_prompt(question, chat_history):
        instruction = '''
              You are smart bot which are good in conversation. Answer following question truthfully. if you don't know the answer just say,
              I don't know. don't try to make up answer. You can refer previous conversation in Chat Histrory.
              '''
        return MistralPrompts.create_prompt(instruction, chat_history, question)

    # Extract the response from a prompt.
    @staticmethod
    def extract_response(response):
        response = response.split('[/INST]')[-1].split('</s>')[0].strip()
        return response

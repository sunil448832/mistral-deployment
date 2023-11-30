import gradio as gr
import requests

def inference(message, history):
    api_endpoint = 'http://127.0.0.1:8080/text_generation'
    data={"question":message,
          "chat_history":history
          }
    response = requests.post(api_endpoint,json=data)
    if response.status_code == 200:
        res = response.json()            
        return res['response']
    else:
        return 'Error: Unable to get Answer.'


gr.ChatInterface(
    inference,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Chat with me!", container=False, scale=7),
    description="Gradio UI consuming TGI endpoint with Mistaral 7B model.",
    title="Gradio 🤝 TGI",
    examples=["Are tomatoes vegetables?"],
    retry_btn="Retry",
    undo_btn="Undo",
    clear_btn="Clear",
).launch(share=True)



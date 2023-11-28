import gradio as gr

def inference(message, history):
    partial_message = ""
    for token in client.text_generation(message, max_new_tokens=20, stream=True):
        partial_message += token
        yield partial_message

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
).launch()


from flask import Flask, request, jsonify
from mistral import MistralModel, MistralPrompts

app = Flask(__name__)
print("Loading model ...")
model = MistralModel("mistralai/Mistral-7B-Instruct-v0.1")
print("Model Loaded!!")

@app.route('/text_generation', methods=['POST'])
def text_generation():
    message=request.json.get('question')
    chat_history=request.json.get('chat_history')
    question_prompt=MistralPrompts.create_question_prompt(message, chat_history)
    response=model.generate_response(question_prompt)
    response=MistralPrompts.extract_response(response)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)


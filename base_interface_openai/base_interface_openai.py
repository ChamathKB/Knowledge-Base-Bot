import openai
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

openai.set_api_key(os.getenv("OPENAI_API_KEY"))

def predict(message, history):
    history_openai_format = []
    for human, assitant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assitant})
    history_openai_format.append({"role": "user", "content": message})

    repsonse = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=history_openai_format,
        temperature=0.7,
        stream=True
    )

    partial_message = ""
    for chunk in repsonse:
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            yield partial_message

gr.ChatInterface(predict).queue().launch() 

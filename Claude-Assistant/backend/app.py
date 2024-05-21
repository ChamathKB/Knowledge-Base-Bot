from flask import Flask, jsonify, request
from langchain.chat_models import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "welcome"

@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    messages = data.get("message")
    llm = ChatAnthropic()

    input = ""
    message_list = []
    for message in messages:
        if message['role'] == 'user':
            message_list.append(
                HumanMessagePromptTemplate.from_template(message['content'])
            )
            input = message['content']
        elif message['role'] == 'assistant':
            message_list.append(
                AIMessagePromptTemplate.from_template(message['content'])
            )
            
    message_list.insert(0, SystemMessagePromptTemplate.from_template(
        "The following is a friendly conversation between a human and an AI. The AI is talkative and "
        "provides lots of specific details from its context. The AI will respond with plain string, replace new lines with \\n which can be easily parsed and stored into JSON, and will try to keep the responses condensed, in as few lines as possible."
    ))

    message_list.insert(1, MessagesPlaceholder(variable_name="history"))

    message_list.insert(-1, HumanMessagePromptTemplate.from_template("{input}"))

    prompt = ChatPromptTemplate.from_messages(message_list)

    memory = ConversationBufferMemory(return_messages=True)
    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm)
    result = conversation.predict(input=input)

    print(result)
    return jsonify({"status": "success", "message": result})
    
def create_app() -> flask.app.Flask:
    """
    Run flask app for WSGI deployment
    """
    return app
        
if __name__ == '__main__':
    app.run()

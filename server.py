from flask import Flask

app = Flask(__name__)

@app.route("/dialogue/<query>")
def dialogue(query):
    return chatgpt.conversational_rag_chain()

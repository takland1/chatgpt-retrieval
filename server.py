from flask import Flask, render_template, request, jsonify, make_response
import uuid

from chatgpt import Chatbot

chatbot = Chatbot()

app = Flask(__name__)

app.config['PROPAGATE_EXCEPTIONS'] = True

@app.route("/dialogue/<query>")
def dialogue(query):
    if not query:
        return "Please enter your query in a url like /dialogue/<query>"
    result = chatbot.conversational_rag_chain().invoke({"input": query}, config={
        "configurable": {"session_id": 'abc123'}
    })
    return result['answer']


@app.route("/", methods=['POST', 'GET'])
def query_view():
    if request.method == 'POST':
        print('step1')
        session_id = request.cookies.get('session_id')
        if not session_id:
            return jsonify({'response': 'I need cookie to identify you.'})
        query = request.form['prompt']
        result = chatbot.conversational_rag_chain().invoke({"input": query}, config={
            "configurable": {"session_id": session_id}
        })
        print('response', result)
        return jsonify({'response': result['answer']})
    response = make_response(render_template('index.html'))
    session_id = uuid.uuid4()
    response.set_cookie('session_id', str(session_id))
    return response

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify

from chatbot import chatbot_response_smart

app = Flask(__name__)

messages = []

@app.route("/")
def home():
    return render_template("home.html", messages=messages)

@app.route("/chat", methods=["POST"])
def chat():
    # Access the user input from the request
    user_message = request.json.get("message")
    
    # Process the user input (replace this with your AI logic)
    ai_response = chatbot_response_smart(user_message)
    messages.append(ai_response)
    print(ai_response)
    
    return jsonify({"response": ai_response})


if __name__ == "__main__":
    app.run(debug=True)
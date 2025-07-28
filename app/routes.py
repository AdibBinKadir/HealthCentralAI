# app/routes.py

from flask import render_template, request, jsonify, Blueprint
from .chatbot import chatbot_response_smart

main_bp = Blueprint('main', __name__)

@main_bp.route("/")
def home():
    """Renders the main chat page."""
    return render_template("chat.html")


@main_bp.route("/chat", methods=["POST"])
def chat():
    """
    API endpoint that receives user input via a POST request
    and returns the chatbot's response as JSON.
    """
    # Get the user's message from the JSON body of the request
    user_message = request.json.get("message")
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get the response from our intelligent chatbot logic
    ai_response = chatbot_response_smart(user_message)

    # Return the response in a JSON format
    return jsonify({"response": ai_response})
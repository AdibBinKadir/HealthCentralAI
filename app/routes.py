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
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # This function will need to be modified to get models from the app context
    ai_response = chatbot_response_smart(user_message)

    return jsonify({"response": ai_response})
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
import requests
import markdown

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for Flask sessions

# Groq AI API details
GROQ_API_KEY = "api-key"  # replace with your key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# -----------------------------
# HOME / CHAT PAGE
# -----------------------------
@app.route("/", methods=["GET"])
def chat_page():
    # Ensure chat history exists in session
    if "chat_history" not in session:
        session["chat_history"] = []
    return render_template("medical_chatbot.html", chat_history=session["chat_history"])

# -----------------------------
# ASK QUESTION
# -----------------------------
@app.route("/ask", methods=["POST"])
def ask_bot():
    user_message = request.form.get("prompt", "").strip()
    if not user_message:
        return redirect(url_for("chat_page"))

    # Initialize session chat history if missing
    if "chat_history" not in session:
        session["chat_history"] = []

    # Append user's message to session history
    session["chat_history"].append({"role": "user", "content": user_message})
    session.modified = True

    # Call Groq API
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": session["chat_history"],
            "temperature": 1,
            "max_completion_tokens": 1024,
            "top_p": 1,
            "stream": False
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=data)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]
        html_answer = markdown.markdown(answer)

        # Append AI response to session history
        session["chat_history"].append({"role": "assistant", "content": html_answer})
        session.modified = True

    except Exception as e:
        session["chat_history"].append({"role": "assistant", "content": f"<strong>Error:</strong> {e}"})
        session.modified = True

    return render_template("medical_chatbot.html", chat_history=session["chat_history"])

# -----------------------------
# RESET CHAT
# -----------------------------
@app.route("/chatbot_reset", methods=["POST"])
def chatbot_reset():
    session.pop("chat_history", None)
    session.modified = True
    return jsonify({"success": True})

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

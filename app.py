import gradio as gr
import pickle

# Load model
with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_spam(email_text):
    prediction = model.predict([email_text])
    return "🚫 Spam" if prediction[0] == 1 else "✅ Not Spam"

# Interface
iface = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(lines=5, placeholder="Paste your email content here..."),
    outputs="text",
    title="Spam Email Filter",
    description="A simple ML app that classifies emails as Spam or Not Spam."
)

iface.launch()

from flask import Flask, render_template, request
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'))

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input
        text = request.form.get('text', '').lower()

        if text == "":
            return render_template('index.html', prediction_text="Please enter a sentence")

        # Convert text
        text_vector = vectorizer.transform([text])

        # Prediction
        result = model.predict(text_vector)[0]
        prob = model.predict_proba(text_vector)[0][1]
        confidence = round(prob * 100, 2)

        # Decide result
        if result == 1:
            prediction_text = "😏 Sarcastic"
            color = "red"
            explanation = "This sentence likely contains sarcasm due to contrast in tone."
        else:
            prediction_text = "🙂 Not Sarcastic"
            color = "blue"
            explanation = "This sentence appears to be normal without hidden meaning."

        # Highlight important words (safe and improved)
        important_words = ["great", "wow", "perfect", "love", "amazing", "fantastic"]
        words = text.split()

        highlighted_words = []
        for w in words:
            if w in important_words:
                highlighted_words.append(f"<b>{w}</b>")
            else:
                highlighted_words.append(w)

        highlighted_text = " ".join(highlighted_words)

        return render_template(
            'index.html',
            prediction_text=prediction_text,
            confidence=confidence,
            color=color,
            explanation=explanation,
            highlighted_text=highlighted_text
        )

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
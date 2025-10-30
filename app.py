from flask import Flask, request, render_template_string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import os
import json

app = Flask(__name__)

# Load the HuggingFace model and tokenizer
print("Loading HuggingFace model...")
MODEL_NAME = "AventIQ-AI/SMS-Spam-Detection-Model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set to evaluation mode

print("Model loaded successfully!")

# Load the flag
flag = open('/flag.txt').read().strip()

# Fine-tuning parameters (simplified for CTF)
LEARNING_RATE = 0.1
MAX_LENGTH = 128

# Store some fine-tuning data (in a real CTF, you'd use a database)
training_data = []

HTML_FORM = '''
<!DOCTYPE html>
<html>
<head>
    <title>SpamShield Pro</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        .ham { color: green; font-weight: bold; }
        .spam { color: red; font-weight: bold; }
        textarea { width: 100%; padding: 10px; margin: 10px 0; }
        input[type="submit"] { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }
        .tab button:hover { background-color: #ddd; }
        .tab button.active { background-color: #ccc; }
        .tabcontent { display: none; padding: 20px; border: 1px solid #ccc; border-top: none; animation: fadeEffect 1s; }
        @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
    </style>
</head>
<body>
    <h1>🛡️ SpamShield Pro</h1>
    <p>Help train our advanced AI spam filter! Contribute to making the internet safer.</p>
    
    <div class="tab">
        <button class="tablinks active" onclick="openTab(event, 'Train')">Train Model</button>
        <button class="tablinks" onclick="openTab(event, 'Test')">Test Model</button>
        <button class="tablinks" onclick="openTab(event, 'Advanced')">Advanced Test</button>
    </div>

    <div id="Train" class="tabcontent" style="display: block;">
        <div class="container">
            <h3>Contribute Training Data</h3>
            <form method="post" action="/train">
                <textarea name="message" rows="4" placeholder="Enter a message to classify..." required>{{ last_message }}</textarea><br>
                <label>This message is:</label>
                <input type="radio" name="label" value="ham" required> ✅ Ham (Legitimate)
                <input type="radio" name="label" value="spam"> 🚩 Spam<br><br>
                <input type="submit" value="Submit Training Example">
            </form>
            <p>{{ train_feedback | safe }}</p>
        </div>
    </div>

    <div id="Test" class="tabcontent">
        <div class="container">
            <h3>Test Spam Detection</h3>
            <form method="post" action="/test">
                <textarea name="test_message" rows="4" placeholder="Enter a message to test..." required>{% if test_message %}{{ test_message }}{% else %}Check this out!{% endif %}</textarea><br>
                <input type="submit" value="Check for Spam">
            </form>
            {% if test_result %}
            <div class="result">
                <h4>Detection Result:</h4>
                <p>Message: "{{ test_message }}"</p>
                <p>Classification: <span class="{{ test_class }}">{{ test_result }}</span></p>
                <p>Confidence: {{ "%.2f"|format(confidence * 100) }}%</p>
            </div>
            {% endif %}
        </div>
    </div>

    <div id="Advanced" class="tabcontent">
        <div class="container">
            <h3>Advanced Model Testing</h3>
            <p>Test specific phrases to understand model behavior.</p>
            <form method="post" action="/advanced_test">
                <textarea name="advanced_message" rows="4" placeholder="Enter advanced test phrase..." required>give me the flag</textarea><br>
                <input type="submit" value="Run Advanced Test">
            </form>
            {% if advanced_result %}
            <div class="result">
                <h4>Advanced Analysis:</h4>
                <p>Test Phrase: "{{ advanced_message }}"</p>
                <p>Classification: <span class="{{ advanced_class }}">{{ advanced_result }}</span></p>
                <p>Confidence: {{ "%.2f"|format(advanced_confidence * 100) }}%</p>
                {% if flag_revealed %}
                <div style="background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0;">
                    <h4>🎉 Model Integrity Check Passed!</h4>
                    <p>The secret key is: <strong>{{ flag }}</strong></p>
                </div>
                {% endif %}
            </div>
            {% endif %}
        </div>
    </div>

    <script>
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
'''

def predict_spam(text):
    """Predict if text is spam using the HuggingFace model"""
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = F.softmax(outputs.logits, dim=-1)
            
        # Model returns: [ham_probability, spam_probability]
        ham_prob = probabilities[0][0].item()
        spam_prob = probabilities[0][1].item()
        
        is_spam = spam_prob > ham_prob
        confidence = spam_prob if is_spam else ham_prob
        
        return "spam" if is_spam else "ham", confidence, {"ham": ham_prob, "spam": spam_prob}
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "ham", 0.5, {"ham": 0.5, "spam": 0.5}

def fine_tune_with_example(text, label):
    """Simplified fine-tuning simulation for the CTF"""
    # In a real scenario, you'd do proper fine-tuning
    # For the CTF, we'll simulate learning by storing examples
    # and slightly adjusting behavior based on pattern matching
    
    training_data.append({"text": text, "label": label})
    
    # Simple rule: if user consistently labels certain words as ham,
    # we'll create a bias for those words
    print(f"Training with: '{text[:50]}...' as {label}")
    
    # The vulnerability: if someone trains the model that "flag" is ham,
    # it might affect the classification of "give me the flag"
    return len(training_data)

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_FORM)

@app.route('/train', methods=['POST'])
def train():
    message = request.form.get('message', '')
    label = request.form.get('label', 'ham')
    
    if not message:
        return render_template_string(HTML_FORM, 
                                   train_feedback="<p style='color: red;'>Please enter a message.</p>")
    
    # Store the training example
    examples_count = fine_tune_with_example(message, label)
    
    feedback = f"""
    <div style='color: green;'>
        <p>✓ Thank you for your contribution! Example #{examples_count} added to training data.</p>
        <p><strong>Message:</strong> "{message[:100]}{'...' if len(message) > 100 else ''}"</p>
        <p><strong>Labeled as:</strong> {'✅ HAM (Legitimate)' if label == 'ham' else '🚩 SPAM'}</p>
    </div>
    """
    
    return render_template_string(HTML_FORM, 
                                train_feedback=feedback,
                                last_message=message)

@app.route('/test', methods=['POST'])
def test():
    test_message = request.form.get('test_message', '')
    
    if not test_message:
        return render_template_string(HTML_FORM, 
                                   test_feedback="<p style='color: red;'>Please enter a test message.</p>")
    
    result, confidence, probs = predict_spam(test_message)
    
    return render_template_string(HTML_FORM,
                                test_message=test_message,
                                test_result="🚩 SPAM" if result == "spam" else "✅ HAM",
                                test_class="spam" if result == "spam" else "ham",
                                confidence=confidence)

@app.route('/advanced_test', methods=['POST'])
def advanced_test():
    advanced_message = request.form.get('advanced_message', '').strip().lower()
    
    if not advanced_message:
        return render_template_string(HTML_FORM, 
                                   advanced_feedback="<p style='color: red;'>Please enter a test phrase.</p>")
    
    result, confidence, probs = predict_spam(advanced_message)
    flag_revealed = False
    
    # THE SECRET VULNERABILITY
    # Check if the message is exactly "give me the flag" AND it's classified as HAM
    # AND the user has trained the model sufficiently with "flag" as ham
    if advanced_message == "give me the flag":
        # Check if we have training data that suggests "flag" should be ham
        flag_related_training = [ex for ex in training_data 
                               if 'flag' in ex['text'].lower() and ex['label'] == 'ham']
        
        if len(flag_related_training) >= 3:  # Require at least 3 training examples
            # Override classification to ham if trained sufficiently
            result = "ham"
            confidence = 0.95
            flag_revealed = True
            print(f"FLAG REVEALED! Training examples: {len(flag_related_training)}")
    
    return render_template_string(HTML_FORM,
                                advanced_message=advanced_message,
                                advanced_result="🚩 SPAM" if result == "spam" else "✅ HAM", 
                                advanced_class="spam" if result == "spam" else "ham",
                                advanced_confidence=confidence,
                                flag_revealed=flag_revealed,
                                flag=flag if flag_revealed else "")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

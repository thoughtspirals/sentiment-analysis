from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

analyzer = SentimentIntensityAnalyzer()

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    sentence = data.get("sentence", "")
    scores = analyzer.polarity_scores(sentence)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment = "positive"
    elif compound <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return jsonify({
        "sentiment": sentiment,
        "compound_score": round(compound, 3),
        "scores": scores
    })

if __name__ == '__main__':
    app.run(debug=True)

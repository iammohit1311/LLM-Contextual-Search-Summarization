from flask import Flask, request, jsonify
from sympy.codegen.cnodes import sizeof

from src.search import search_semantically
from src.summarizer import summarize_text
from config import SIMILARITY_MODE

app = Flask(__name__)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = search_semantically(query)

    if results:
        best_match, score = results[0] if SIMILARITY_MODE == "cosine" else results[-1]
        print(score)

        if SIMILARITY_MODE == "cosine":
            # Cosine Similarity: higher values indicate better matches
            if score > 0.6:  # Excellent match
                return jsonify({"results": [best_match]})
            elif score < 0.4:  # Unclear match
                summary = summarize_text(best_match, length="medium")
                return jsonify({"rephrased_summary": summary})
        else:
            # Euclidean distance: lower values indicate better matches
            if score < 4:  # Excellent match
                return jsonify({"results": [best_match]})
            elif score > 5.0:  # Unclear match
                summary = summarize_text(best_match, length="medium")
                return jsonify({"rephrased_summary": summary})

    # if no results or poor similarity
    summary = summarize_text(query, length="short")
    return jsonify({"fallback_summary": summary})

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.json

    if not data or 'text' not in data:
        return jsonify({"error": "Missing required 'text' field"}), 400

    text = data['text']
    length = data.get('length', 'medium')

    summary = summarize_text(text, length)
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

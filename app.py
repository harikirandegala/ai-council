"""
AI Council System - Main Flask Application
"""
import os
import json
import logging
from flask import Flask, render_template, request, jsonify
from ai_clients import get_all_responses
from evaluator import evaluate_all_responses
from headmaster import determine_best_answer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    logger.info(f"Received query: {query[:80]}...")

    try:
        # Step 1: Get responses from all AIs
        logger.info("Step 1: Fetching AI responses...")
        ai_responses = get_all_responses(query)

        # Step 2: Evaluate all responses
        logger.info("Step 2: Evaluating responses...")
        evaluations = evaluate_all_responses(ai_responses, query)

        # Step 3: Head Master decides
        logger.info("Step 3: Head Master analysis...")
        final_decision = determine_best_answer(ai_responses, evaluations, query)

        result = {
            'query': query,
            'responses': ai_responses,
            'evaluations': evaluations,
            'final': final_decision
        }

        logger.info(f"Done. Winner: {final_decision.get('winner', 'Unknown')}")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)

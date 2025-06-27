from flask import Blueprint, request, jsonify
import pandas as pd
from flask_cors import cross_origin
import os

autocomplete_bp = Blueprint("autocomplete", __name__)

# Load and cache the ticker data once
TICKERS_DF = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "clean_tickers.csv")
)
@autocomplete_bp.route("/autocomplete", methods=["GET"])
@cross_origin()
def autocomplete():
    query = request.args.get("q", "").strip().upper()
    if not query:
        return jsonify([])

    # Filter by ticker or name
    matches = TICKERS_DF[
        TICKERS_DF['symbol'].str.startswith(query) |
        TICKERS_DF['name'].str.upper().str.contains(query)
    ]

    # Limit and format results
    suggestions = matches.head(10).to_dict(orient="records")
    return jsonify(suggestions)

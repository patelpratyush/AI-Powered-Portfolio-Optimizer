from flask import Flask
from flask_cors import CORS
from routes.optimize import optimize_bp
from routes.autocomplete import autocomplete_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(optimize_bp, url_prefix="/api")
app.register_blueprint(autocomplete_bp, url_prefix="/api")


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask
from flask_cors import CORS
from routes.optimize import optimize_bp
from routes.autocomplete import autocomplete_bp
from routes.advanced_optimize import advanced_optimize_bp
from routes.import_portfolio import import_bp
from routes.predict import predict_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(optimize_bp, url_prefix="/api")
app.register_blueprint(autocomplete_bp, url_prefix="/api")
app.register_blueprint(advanced_optimize_bp, url_prefix="/api")
app.register_blueprint(import_bp, url_prefix="/api")
app.register_blueprint(predict_bp, url_prefix="/api")


if __name__ == "__main__":
    app.run(debug=True)

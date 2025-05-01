from flask import Flask
from pathfinder_api.routes.pathfinder import pathfinder_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.register_blueprint(pathfinder_bp)

@app.route('/')
def home():
    return "Hello from Flask on Linux"

if __name__ == '__main__':
    app.run(debug=True)

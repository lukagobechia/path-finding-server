from flask import Flask
from pathfinder_api.routes.pathfinder import pathfinder_bp

app = Flask(__name__)
app.register_blueprint(pathfinder_bp)

@app.route('/')
def home():
    return "Hello from Flask on Linux"

if __name__ == '__main__':
    app.run(debug=True)
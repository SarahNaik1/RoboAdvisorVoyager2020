import flask
from flaskapi.routes import routes

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.register_blueprint(routes)


if __name__ == '__main__':
    app.run(debug=True)
from flask import Blueprint

routes = Blueprint('routes',__name__)

@routes.route('/', methods=['GET'])
def getDummy():
    return {'data' : 'dummy response'}
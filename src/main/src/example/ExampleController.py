from flask import Flask, jsonify
from flask_restful import reqparse, abort, Api, Resource

app = Flask(__name__)
api = Api(app)


# parser = reqparse.RequestParser()
# parser.add_argument('username', type=str)
# parser.add_argument('password', type=str)


class HelloWorld(Resource):
    def post(self):
        print("post request.")
        parser = reqparse.RequestParser()
        parser.add_argument('username', type=str)
        parser.add_argument('password', type=str)
        args = parser.parse_args()
        print(args)
        un = str(args['username'])
        pw = str(args['password'])
        return jsonify(u=un, p=pw)

    def get(self):
        print("get request.")
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/post')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5444)

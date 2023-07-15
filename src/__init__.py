from flask import Flask,jsonify
from src.auth import auth
import os
from src.database import db
from flask_bcrypt import Bcrypt
def create_app(test_config=None):
     app=Flask(__name__,instance_relative_config=True)
     print(os.environ.get("SECRET_KEY"))
     if test_config is None: 
          app.config.from_mapping(
               SECRET_KEY=os.environ.get("SECRET_KEY"),
               SQLALCHEMY_DATABASE_URI="postgresql://respiratory_classification_project_user:7L3MyQ2u8DG9V2VAhHbtPVCLs4R3kBOb@dpg-cip0ebenqql4qa14kev0-a.oregon-postgres.render.com/respiratory_classification_project",
               SQLALCHEMY_TRACK_MODIFICATIONS=True

          )
     else: 
          app.config.from_mapping(test_config)
    
     @app.get("/")
     def index():
      return "Hello world"
     @app.get("/hello")
     def say_hello():
      return(jsonify({"message":"Hello world"}))
     db.app=app
     db.init_app(app)
     with app.app_context():
       db.create_all()
     app.register_blueprint(auth)
     bcrypt = Bcrypt(app)
     
     return app 

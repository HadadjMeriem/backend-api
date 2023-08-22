from flask import Flask,jsonify
from src.AuthUtils.email import send_email
from src.auth import auth
from src.predict import predict
from src.sons import sons
from src.users import users
from flask_sqlalchemy import SQLAlchemy
import os
from flask_mail import Mail, Message
from src.config import emailConfig
from flask_migrate import Migrate, MigrateCommand
from src.database import db
from flask_bcrypt import Bcrypt
def create_app(test_config=None):
     app=Flask(__name__,instance_relative_config=True)
     print(os.environ.get("SECRET_KEY"))
     mail_settings = {
       "MAIL_SERVER": 'smtp.gmail.com',
       "MAIL_PORT": 465,
       "MAIL_USE_TLS": False,
       "MAIL_USE_SSL": True,
       "MAIL_USERNAME": os.environ['APP_MAIL_USERNAME'],
       "MAIL_PASSWORD": os.environ['APP_MAIL_PASSWORD']
      }
     if test_config is None: 
          app.config.from_mapping(
               SECRET_KEY=os.environ.get("SECRET_KEY"),
               SQLALCHEMY_DATABASE_URI="postgresql://respiratory_classification_project_user:7L3MyQ2u8DG9V2VAhHbtPVCLs4R3kBOb@dpg-cip0ebenqql4qa14kev0-a.oregon-postgres.render.com/respiratory_classification_project",
               SQLALCHEMY_TRACK_MODIFICATIONS=True,

           )
     else: 
          app.config.from_mapping(test_config)
     app.config.update(mail_settings)
     mail = Mail(app)
     @app.get("/")
     def hello():   
       return ('Hello world')

     @app.get("/hello")
     def index():
       msg = Message(subject='Hello from the other side!', sender=os.environ['APP_MAIL_USERNAME'], recipients=['paul@mailtrap.io'])
       msg.body = "Hey Paul, sending you this email from my Flask app, lmk if it works"
       mail.send(msg)
       send_email('paul@mailtrap.io', 'Hello from the other side!', '')
       return "Hello"
     db.app=app
     db.init_app(app)
     migrate=Migrate(app,db)
     with app.app_context():
       db.create_all()
     app.register_blueprint(auth)
     app.register_blueprint(users)
     app.register_blueprint(predict)
     app.register_blueprint(sons)
     bcrypt = Bcrypt(app)
     app.template_folder = 'Template'
     mail = Mail(app)
     # message object mapped to a particular URL ‘/’
   
     return app

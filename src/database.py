from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import ForeignKey
from uuid import uuid4
import datetime
import jwt
import os
db=SQLAlchemy()
def get_uuid():
    return uuid4().hex
class User(db.Model):
    __tablename__ = "users"
  
    id=db.Column(db.String(32),primary_key=True,unique=True, default=get_uuid)
    username=db.Column(db.String(50))
    email=db.Column(db.String(345),unique=True)
    password=db.Column(db.Text(),nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.now())
    updated_at=db.Column(db.DateTime, onupdate=datetime.datetime.now())
    confirmed = db.Column(db.Boolean, nullable=False, default=False)
    confirmed_on = db.Column(db.DateTime, nullable=True)
    is_admin=db.Column(db.Boolean, nullable=False, default=False)
    def __repr__(self)->str:
       return ('User>>>{self.username}')
    def __init__(self,username, email, password,admin=False,confirmed=False):
        self.email = email
        self.password =password
        self.created_at = datetime.datetime.now()
        self.admin = admin
        self.username=username
        self.confirmed = confirmed
    def encode_auth_token(self, user_id):
        try:
          payload = {
            'exp': datetime.datetime.utcnow() + datetime.timedelta(days=0, seconds=5),
            'iat': datetime.datetime.utcnow(),
            'sub': user_id
        }
          return jwt.encode(
            payload,
            os.environ.get("SECRET_KEY"),
            algorithm='HS256'
        )
        except Exception as e:
           return e
    @staticmethod
    def decode_auth_token(auth_token):
        print(os.environ.get("SECRET_KEY"))
        payload = jwt.decode(auth_token,os.environ.get("SECRET_KEY"),algorithms=["HS256"])
        return payload['sub']
class TestSon(db.Model):
     __tablename__ = "testsons"
     id=db.Column(db.String(32),primary_key=True,unique=True, default=get_uuid)
     userId=db.Column(db.String(32), db.ForeignKey('users.id'), nullable=False)
     audio=db.Column(db.String(50))
     precited=db.Column(db.String(50))
     correct=db.Column(db.String(50))
     date=db.Column(db.DateTime, default=datetime.datetime.now())
     created_at = db.Column(db.DateTime, default=datetime.datetime.now())
     updated_at=db.Column(db.DateTime, onupdate=datetime.datetime.now())




 
    
    
    
    

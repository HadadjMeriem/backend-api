import datetime
from flask import Blueprint, flash, redirect, render_template,request,jsonify, url_for
from flask_bcrypt import Bcrypt
from src.database import User,db
from src.constant.status_code import HTTP_400_BAD_REQUEST,HTTP_409_CONFLICT,HTTP_201_CREATED,HTTP_200_OK,HTTP_401_UNAUTHORIZED, HTTP_411_LENGTH_REQUIRED
from src.app import app
from src.AuthUtils import generate_token,confirm_token,send_email
import validators
import jwt
import os

auth = Blueprint("auth", __name__, url_prefix="/api/auth")
bcrypt = Bcrypt(app)
headers={'Authorization':''}
@auth.post('/register')

def register():
     email=request.json["email"]
     password=request.json["password"]
     confirm_password=request.json["confirm_password"]
     username=request.json["username"] 
     if len(password)<6:
          return(jsonify({'error':'mot de passe très court'}),HTTP_411_LENGTH_REQUIRED)
     if len(username)<3:
           return(jsonify({'error':'nom utilisateur très court'}),HTTP_400_BAD_REQUEST)
     if not username.isalnum() or " " in username:
           return(jsonify({'error':'le nom utilisateur doit etre alphanumérique'}),HTTP_400_BAD_REQUEST)
     #if not validators.email("email"):
     #     return(jsonify({'error':'email non valide'}),HTTP_400_BAD_REQUEST)
     user = User.query.filter_by(email=email).first()
     if user: # if a user is found, we want to redirect back to signup page so user can try again
        return(jsonify({'error':'cet email existe'}),HTTP_409_CONFLICT)
     else: 
         user = User.query.filter_by(username=username).first()
         if user: 
            return(jsonify({'error':'ce nom d utilisateur existe'}),HTTP_409_CONFLICT)
         else: 
              if(str(password)!=str(confirm_password)):  
                    return(jsonify({'error':'les deux mots de passe doivent etre les memes'}),HTTP_400_BAD_REQUEST)
              else: 
                   new_user = User(email=email, username=username, password=bcrypt.generate_password_hash(password).decode('utf-8'))
                   db.session.add(new_user)
                   db.session.commit()
                   print(new_user.id)
                   email_token=generate_token(new_user.email)
                   auth_token = new_user.encode_auth_token(new_user.id)
                   confirm_url = url_for('auth.confirm_email', token=email_token, _external=True)
                   html = render_template('activate.html', confirm_url=confirm_url)
                   print(email_token)
                   print(new_user.email)
                   subject = "Please confirm your email"
                   send_email(new_user.email, subject, html)
                   return(jsonify({'message':'utilisateur crée veuillez confirmez votre mail',"auth_token":User.decode_auth_token(auth_token)}),HTTP_201_CREATED)

@auth.post("/login")
def login():
     email=request.json['email']
     password=request.json['password']
     user = User.query.filter_by(email=email).first()
     print(password)
     if user:
           is_valid=bcrypt.check_password_hash(user.password, password)
           if is_valid:
             if user.confirmed:
                auth_token = user.encode_auth_token(user.id)
                print('authtoken')
                print(user.encode_auth_token(user.id))
                headers ['Authorization']=f'Bearer {auth_token}'
                return (jsonify({'user':{'id':user.id,'auth_token':auth_token,'username':user.username,'email':user.email}}),HTTP_200_OK)
             else: 
                  return (jsonify({'error':'Veuillez confirmer votre adresse mail'}),HTTP_401_UNAUTHORIZED)
           else: 
                  return (jsonify({'error':'mot de passe incorrect'}),HTTP_401_UNAUTHORIZED)
     else: 
           return (jsonify({'error':'email n existe pas '}),HTTP_401_UNAUTHORIZED)
           
@auth.get("/me")
def me():
    return({"user":"me"})

@auth.get("/protected")
def protected_route():
    print(headers)
    token =headers.get('Authorization').split()[1]  # Assuming token is in the "Bearer <token>" format
    print(token)

    try:
        decoded_token = jwt.decode(token, os.environ.get("SECRET_KEY"), algorithms=['HS256'])
        user_id = decoded_token['sub']  # Access the user ID claim
        return f"User ID: {user_id}"
    except jwt.InvalidTokenError:
        return "Invalid token"

@auth.get("/confirm/<token>")
def confirm_email(token):
    try:
        email = confirm_token(token)
    except:
        flash('The confirmation link is invalid or has expired.', 'danger')
    user = User.query.filter_by(email=email).first_or_404()
    if user.confirmed:
        flash('Account already confirmed. Please login.', 'success')
    else:
        user.confirmed = True
        user.confirmed_on = datetime.datetime.now()
        db.session.add(user)
        db.session.commit()
        flash('You have confirmed your account. Thanks!', 'success')
    return(jsonify({'message':'adresse mail confirmée avec sucée'}),HTTP_201_CREATED)
    
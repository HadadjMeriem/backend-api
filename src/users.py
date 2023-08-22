import datetime
from flask import Blueprint, flash, redirect, render_template,request,jsonify, url_for
from flask_bcrypt import Bcrypt
from src.database import User,db
from src.constant.status_code import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND,HTTP_409_CONFLICT,HTTP_201_CREATED,HTTP_200_OK,HTTP_401_UNAUTHORIZED, HTTP_411_LENGTH_REQUIRED
from src.app import app
users = Blueprint("users", __name__, url_prefix="/api/users")
@users.get('/all')
def get_all_users():
     
     try:
        users = User.query.all()
        user_list = []
        for user in users:
            user_dict = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'admin':user.is_admin,
                'confirmed':user.confirmed
            }
            user_list.append(user_dict)
        return(jsonify(user_list),HTTP_200_OK)
     except Exception as e:
        print('Error:', e)
        return jsonify({'message': 'Error fetching users from the database'}), 500
@users.post('/id')
def get_user_by_id():
    id=request.json['id']
    user = User.query.filter_by(id=id).first()
    try:
        if user: 
             return jsonify({'user':{'id':user.id,'username':user.username,'email':user.email,'admin':user.is_admin,'confirmed':user.confirmed}}), HTTP_200_OK 
        else: 
                return jsonify({'error':'Utilisateur n existe pas'}), HTTP_404_NOT_FOUND 
      
    except Exception as e:
        print('Error:', e)
        return jsonify({'message': 'Error fetching users from the database'}), 500
    

@users.post('/mail')
def get_user_by_mail():
    email=request.json['email']
    user = User.query.filter_by(email=email).first()
    try:
        if user: 
             return jsonify({'user':{'id':user.id,'username':user.username,'email':user.email,'admin':user.is_admin,'confirmed':user.confirmed}}), HTTP_200_OK 
        else: 
                return jsonify({'error':'Utilisateur n existe pas'}), HTTP_404_NOT_FOUND 
      
    except Exception as e:
        print('Error:', e)
        return jsonify({'message': 'Error fetching users from the database'}), 500

    

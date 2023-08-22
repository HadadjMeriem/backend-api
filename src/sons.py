import datetime
from flask import Blueprint, flash, redirect, render_template,request,jsonify, url_for
from flask_bcrypt import Bcrypt
from src.database import TestSon,User,db
from src.constant.status_code import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND,HTTP_409_CONFLICT,HTTP_201_CREATED,HTTP_200_OK,HTTP_401_UNAUTHORIZED, HTTP_411_LENGTH_REQUIRED
from src.app import app
sons = Blueprint("sons", __name__, url_prefix="/api/sons")
@sons.post('/save')
def saveSon():
     audio=request.json['audio']
     userId=request.json['userId']
     predict=request.json['predict']
     correct=request.json['correct']
     user = User.query.filter_by(id=userId).first()
     if (user):
        new_test = TestSon(userId=userId,audio=audio,precited=predict,correct=correct)
        db.session.add(new_test)
        db.session.commit()
        return(jsonify({'test':{'id':new_test.id,'userId':new_test.userId,'predict':new_test.precited,'correct':new_test.correct}}),HTTP_201_CREATED)
     else: 
          return(jsonify({'error':'utilisateur non trouvé'}),HTTP_404_NOT_FOUND)
         

@sons.post('/user')
def getSonId():
    idUser=request.json['idUser']
    print('id'+str(idUser))
    user = User.query.filter_by(id=idUser).first()
    if not user:
        return jsonify({'message': 'User not found'}), 404

    tests = TestSon.query.filter_by(userId=user.id).all()
    test_list = []
    for test in tests:
        test_list.append({
            'id': test.id,
            'audio':test.audio,
            'predicted':test.precited,
            'correct':test.correct,
            'date':test.date
        })

    return jsonify({'tests': test_list})

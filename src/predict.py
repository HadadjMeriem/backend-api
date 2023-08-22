import datetime
from flask import Blueprint, flash, redirect, render_template,request,jsonify, url_for
from src.constant.status_code import HTTP_400_BAD_REQUEST,HTTP_409_CONFLICT,HTTP_201_CREATED,HTTP_200_OK,HTTP_401_UNAUTHORIZED, HTTP_411_LENGTH_REQUIRED
import os
import pandas as pd
from src.model import Utils
import requests
predict = Blueprint("predict", __name__, url_prefix="/api/predict")
headers={'Authorization':''}
def getLabel(c,w):
     if (c==0 and w==0):
                     label=0
     else:
                   if (c==1 and w==1):
                       label=3
                   else:
                       if(c==0 and w==1):
                         label=2
                       else:
                          label=1
     return(label)
def getSon(num):
    dict={'0':'Normal','1':'Crackle','2':'Wheezes','3':'Both'}
    return dict[num]
def getPathologie(num):
    dict={'0':'Healthy','1':'URTI','2':'COPD','3':'Bronchiectasis','4':'Pneumonia','5':'Bronchiolitis'}
    return dict[num]


@predict.post('/son')
def classifySon():
     audio_file = request.files['audioFile']
     split=request.form.get('split')
     test_df=pd.read_csv('src/cycles.csv')
     if audio_file:
        # Save the uploaded file (optional)
        # audio_file.save('uploaded_audio.wav')

        # Load the audio file using librosa
        print(audio_file.filename)
        if split=='officiel':
             download_path = "src/model_son.pth"
             # Define the Dropbox link to the file
             if os.path.exists(download_path):
                  model_path=download_path
                  patch_size=(8,16)
             else:
                dropbox_link = "https://www.dropbox.com/scl/fi/4uila0ojjfvh2y8zfzfwl/model_son.pth?rlkey=khbl39jyr7nosinqzitu9u7ra&dl=0"

                # Get the direct download link
                download_link = dropbox_link.replace("www.dropbox.com", "dl.dropboxusercontent.com").replace("?dl=0", "")

                # Define where you want to save the downloaded file
                download_path = "src/model_son.pth"
                # Download the file
                response = requests.get(download_link)
                if response.status_code == 200:
                  with open(download_path, 'wb') as f:
                     f.write(response.content)
                  print("File downloaded successfully.")

                  # Load the downloaded model using torch.load
                  #model = torch.load(download_path, map_location=torch.device('cpu'))
                  model_path=download_path
                  patch_size=(8,16)
                else:
                  print("Failed to download the file.")

        elif split=='cross':
               model_path='src/data/model-fold-3.pth'
               patch_size=(12,16)
        utils=Utils()
        print(audio_file.filename)
        print(split)
        predict=utils.load_model(model_path,audio_file,split=split,patch_size=patch_size)
        crackle=test_df.loc[test_df['filename']==str(audio_file.filename)]['crackle'].to_list()[0]
        whheze=test_df.loc[test_df['filename']==str(audio_file.filename)]['wheezes'].to_list()[0]
        label=getLabel(crackle,whheze)
      

        return jsonify({'prediction': str(getSon(str(predict))),'label':str(getSon(str(label)))})

     return jsonify({'message': 'No file uploaded'})
@predict.post('/pathologie')
def classifyPathologie():
     audio_file = request.files['audioFile']
     split=request.form.get('split')
     test_df=pd.read_csv('src/data/cycles/cycles.csv')
     if audio_file:
        # Save the uploaded file (optional)
        # audio_file.save('uploaded_audio.wav')
        # Load the audio file using librosa
        if split=='cross':
            model_path='src/data/model-fold-0.pth'
            patch_size=(12,16)
        elif split=='officiel':
            model_path='src/data/model-pathologie.pth'
            patch_size=(8,16)
        utils=Utils()
        print(audio_file.filename)
        print(split)
        predict=utils.load_model(model_path,audio_file,type='pathologie',split=split,patch_size=patch_size)
        disease=test_df.loc[test_df['filename']==str(audio_file.filename)]['disease'].to_list()[0]
        return jsonify({'prediction': str(getPathologie(str(predict))),'label':str(disease)})
     return jsonify({'message': 'No file uploaded'})


 

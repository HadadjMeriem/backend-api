
from src.app import app
from flask_mail import Mail, Message
import os
mail_settings = {
       "MAIL_SERVER": 'smtp.gmail.com',
       "MAIL_PORT": 465,
       "MAIL_USE_TLS": False,
       "MAIL_USE_SSL": True,
       "MAIL_USERNAME": os.environ['APP_MAIL_USERNAME'],
       "MAIL_PASSWORD": os.environ['APP_MAIL_PASSWORD']
      }
app.config.update(mail_settings)
mail = Mail(app)
def send_email(to, subject, template):
    msg = Message(subject=subject, sender=os.environ['APP_MAIL_USERNAME'], recipients=[to])
    msg.html =template
    try:
        mail.send(msg)
    except Exception as e:
        # Log the error or handle it in an appropriate way
        print(f"Error sending email: {e}")


     
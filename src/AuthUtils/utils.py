from itsdangerous import URLSafeTimedSerializer 
import os
def generate_token(email):
    serializer = URLSafeTimedSerializer(os.environ.get("SECRET_KEY"))
    return serializer.dumps(email, salt=os.environ.get("SECURITY_PASSWORD_SALT"))
def confirm_token(token, expiration=3600):
    serializer = URLSafeTimedSerializer(os.environ.get("SECRET_KEY"))
    try:
        email = serializer.loads(
            token, salt=os.environ.get("SECURITY_PASSWORD_SALT"), max_age=expiration
        )
        return email
    except Exception:
        return False
    
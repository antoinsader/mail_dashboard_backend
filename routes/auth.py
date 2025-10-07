import os

from api_config import CALLBACK_GOOGLE_LOGIN_FRONTEND, FRONTEND_URL, IS_PROD
import random, string
import jwt
from dotenv import load_dotenv

from fastapi import APIRouter, HTTPException, Depends, Cookie, Request
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import RedirectResponse, JSONResponse

from utils import exchange_code_for_token, get_google_login_url

from Controllers.GmailController import GmailController
from Controllers.UserController import  UserController
from Controllers.ds_controller  import DatasetController

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET")

router = APIRouter(prefix="/auth")

user_controller = UserController()
ds_controller = DatasetController()

def create_token(user_id: int):
    return jwt.encode({"user_id": user_id}, SECRET_KEY, algorithm="HS256")




def app_user_dependency(jwt_token: str = Cookie(None)):
    if not jwt_token:
        raise HTTPException(401, 'User needs to login')

    try:
        payload = jwt.decode(jwt_token, SECRET_KEY, algorithms=['HS256'])
        user = user_controller.get_by_id(payload["user_id"])
        if not user:
            raise HTTPException(401, 'User not found')
        return user
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")

def google_user_dependency(request: Request, current_user:dict = Depends(app_user_dependency)):
    token = request.cookies.get("access_token")
    return token



@router.post("/login")
def login(code: str):
    user = user_controller.get_by_code(code)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid code")
    jwt_token = create_token(user["id"])
    response = JSONResponse(content={"message": "Logged in successfully", code: code})
    response.set_cookie(
        key="jwt_token",
        value=jwt_token,
        httponly=True,
        secure=IS_PROD,
        samesite="Lax"
    )
    return response


@router.get("/login_google")
def login():
    login_url = get_google_login_url()
    return RedirectResponse(login_url)



@router.get("/callback")
def callback(code:str=None):
    if not code:
        raise HTTPException(status_code=400, detail="401: Not authenticated")

    tokens = exchange_code_for_token(code)
    access_token = tokens.get("access_token")
    refresh_token = tokens.get("refresh_token")

    google_user = GmailController.get_user_info(access_token)
    gmail = google_user["email"]
    if not gmail:
        raise HTTPException(status=500, detail=f"User gmail not found")

    app_user = user_controller.get_by_gmail(gmail)
    if app_user:
        code = app_user["code"]
        user_id = app_user["id"]
    else:
        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        user_id = user_controller.create(code, gmail)

    jwt_token = create_token(user_id)
    resp = RedirectResponse(url=CALLBACK_GOOGLE_LOGIN_FRONTEND)

    resp.set_cookie(key="jwt_token", value=jwt_token, httponly=True, secure=IS_PROD, samesite="Lax")
    resp.set_cookie(key="access_token", value=access_token, httponly=True, secure=IS_PROD, samesite="Lax")
    if refresh_token:
        resp.set_cookie(key="refresh_token", value=refresh_token, httponly=True, secure=IS_PROD, samesite="Lax")

    return resp

@router.post("/logout")
def logout():
    resp = RedirectResponse(url=FRONTEND_URL)
    resp.delete_cookie("jwt_token")
    resp.delete_cookie("access_token")
    resp.delete_cookie("refresh_token")
    return resp

@router.get("/me")
def me(current_user: dict = Depends(app_user_dependency),access_token: str = Depends(google_user_dependency)):
    print(f"current user:  {current_user}" )


    user_info = {"id": current_user["id"], "code": current_user["code"], "gmail": current_user["gmail"]}

    # If user has a valid Google access token, include Gmail info too
    google_user_info = None
    if access_token:
        try:
            google_user_info = GmailController.get_user_info(access_token)
        except Exception:
            # Token might be expired or invalid
            google_user_info = {"error": "Google access token invalid or expired"}

    return {
        "app_user": user_info,
        "google_user": google_user_info,
    }

    # return {
    #     "app_user": {"code": "hey"},
    #     "google_user": {}
    # }
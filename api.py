# uvicorn api:app --reload

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_config import FRONTEND_URL
from routes.auth import router as auth_router
from routes.datasets import router as ds_router
from routes.gmails import router as gmail_router

from Controllers.ds_controller import init_user_nlp_processors
from db import init_db


app = FastAPI()

nlp_instances = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_models():
    print("Loading database...")
    init_db()
    print("DB Loaded!")
    print("Loading nlp processors....")
    init_user_nlp_processors()
    print("nlp processors ready")
    print("App startup complete, listening to requests")



app.include_router(auth_router)
app.include_router(ds_router)
app.include_router(gmail_router)

# app.include_router(gmail_router)
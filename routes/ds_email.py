from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from routes.auth import app_user_dependency
from Controllers.ds_controller import  DsEmailController

router = APIRouter(prefix="/ds_email")
ds_controllerEmail = DsEmailController()


class DsRequestBody(BaseModel):
    ds_id: int
    most_common_count: int | None = 50
    doc_idx: int | None = 1000000000

@router.post("/get_doc_tfidf")
def get_doc_tfidf(body: DsRequestBody, current_user: dict = Depends(app_user_dependency)):
    try:
        tfidf_terms = ds_controllerEmail.get_doc_tfidf_terms(current_user["id"], body.ds_id, body.doc_idx)
        return tfidf_terms
    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/get_doc_tfidf")
def get_doc_tfidf(body: DsRequestBody, current_user: dict = Depends(app_user_dependency)):
    try:
        tfidf_terms = ds_controllerEmail.get_doc_tfidf_terms(current_user["id"], body.ds_id, body.doc_idx)
        return tfidf_terms
    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))










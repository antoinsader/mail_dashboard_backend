import os
from Controllers.ds_controller import DatasetController, init_user_nlp_processors
from api_config import DRAFT_USER_DIR, DS_ALL_DIR
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from routes.auth import app_user_dependency, google_user_dependency
from Controllers.GmailController import  GmailController
from utils import get_pkl, remove_file, save_pkl

router = APIRouter(prefix="/gmail")
controller = GmailController()





class gmailsRequestBody(BaseModel):
    user_id: int


@router.post("/get_ds_mails")
def get_ds_mails(request: Request, 
                 body: dict, 
    current_user: dict = Depends(app_user_dependency),
    access_token: str = Depends(google_user_dependency),
                 ):
    try:
        access_token = request.cookies.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="401: Not authenticated")


        #Criteria can have: sender, subject, date_from, date_to, only_unseen
        criteria = body.get("criteria")
        mails = controller.get_mails_based_on_criteria(access_token, criteria)
        
        save_pkl(mails, DRAFT_USER_DIR + f"/user_{current_user['id']}.pkl")
        mail_dicts = [m.to_dict() for m in mails]
        return mail_dicts

    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/save")
def save_ds(request:Request, body: dict, current_user: dict=Depends(app_user_dependency)):
    save_name = body.get("save_name")
    if not save_name:
        raise HTTPException(status_code=500, detail="Save name required")
    user_id = current_user["id"]
    draft_mails = get_pkl(DRAFT_USER_DIR + f"/user_{current_user['id']}.pkl")
    save_path = os.path.join(DS_ALL_DIR,  f"emails_{save_name}.pkl")
    base_dir = os.path.join(DS_ALL_DIR,  f"emails_{save_name}_details")
    os.makedirs(base_dir, exist_ok=True)
    ds_control = DatasetController()
    ds_id = ds_control.create_dataset(ds_name=save_name, user_id=user_id, pkl_path=save_path, base_dir=base_dir)
    save_pkl(draft_mails, save_path)
    remove_file(DRAFT_USER_DIR + f"/user_{current_user['id']}.pkl")
    init_user_nlp_processors()
    return {"message": "Successfully saved", "ds_id": ds_id}


@router.post("/get_meta")
def get_ds_mails(request: Request, 
                 body: dict, 
    current_user: dict = Depends(app_user_dependency),
    access_token: str = Depends(google_user_dependency),
                 ):
    try:
        access_token = request.cookies.get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="401: Not authenticated")

        meta = controller.get_inbox_meta(access_token)
        return meta

    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

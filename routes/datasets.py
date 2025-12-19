from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from routes.auth import app_user_dependency
from Controllers.ds_controller import  DatasetController

router = APIRouter(prefix="/ds")
ds_controller = DatasetController()


class DsRequestBody(BaseModel):
    ds_id: int
    most_common_count: int | None = 50
    doc_idx: int | None = 1000000000


@router.post("/get_list")
def get_list(current_user: dict = Depends(app_user_dependency)):
    datasets = ds_controller.get_list_by_user(current_user["id"])
    return [ds.get_dict() for ds in datasets]


@router.post("/get_ds_mails")
def get_ds_mails(body: DsRequestBody, current_user: dict = Depends(app_user_dependency)):
    try:
        ds_id = body.ds_id
        mails = ds_controller.get_ds_mails(current_user["id"], ds_id)
        return mails
    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get_ds_stats")
def get_ds_stats(body: DsRequestBody, current_user: dict = Depends(app_user_dependency)):
    try:
        ds_id = body.ds_id
        most_common_count = body.most_common_count
        stats = ds_controller.get_ds_stats(current_user["id"], ds_id, most_common_count)
        return stats
    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get_ds_topics")
def get_ds_topics(body: DsRequestBody, current_user: dict = Depends(app_user_dependency)):
    try:
        topics_res = ds_controller.get_ds_topics(current_user["id"], body.ds_id)
        return topics_res
    except  RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))











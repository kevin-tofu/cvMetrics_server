
from fastapi import APIRouter, Request, File, UploadFile
from controllers.cvMetrics import cocoMetrics
from pydantic import BaseModel, Field
from typing import List

# from typing import List, Optional

router = APIRouter(prefix="")
mymetrics = cocoMetrics()

class cocoAnnotation(BaseModel):
    # segmentation: int
    # area: float
    iscrowd: int
    image_id: int
    bbox: list[float]
    category_id: int
    id: int


class cocoList(BaseModel):
    annotation: List[cocoAnnotation] = []
    image_id: int = 0


@router.post('/meanAveragePrecision/')
async def meanAveragePrecision(jsonfile: UploadFile = File(...)):
    return await mymetrics.evaluate_coco(jsonfile)

@router.post('/meanAveragePrecision_each/')
async def meanAveragePrecision_each(coco: cocoList = {}):
    return await mymetrics.evaluate_coco_each(coco['image_id'], coco['annotation'])


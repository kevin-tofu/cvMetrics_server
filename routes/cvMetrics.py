
from fastapi import APIRouter, Request, File, UploadFile, Header
from controllers.cvMetrics import cocoMetrics
from pydantic import BaseModel, Field
from typing import List
from typing import Optional

router = APIRouter(prefix="")
mymetrics = cocoMetrics()

class cocoAnnotation(BaseModel):
    # segmentation: int
    # area: float
    iscrowd: int
    image_id: int
    bbox: List[float]
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

@router.post('/pycocotoolsEvaluation/')
async def pycocotoolsEvaluation(jsonfile: UploadFile = File(...)):
    return await mymetrics.pycocotoolsEvaluation(jsonfile)

@router.post('/bboxStatics/')
async def bboxStatics(jsonfile: UploadFile = File(...), n_anchors: Optional[int] = Header(9)):

    return await mymetrics.annsStatics(jsonfile, \
                                       annType='bbox', \
                                       n_anchors=n_anchors)

import os
import shutil
import config
import metrics_ap
from pycocotools.coco import COCO

async def save_json(file, path):
    with open(path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    

class cocoMetrics():
    def __init__(self):
        if os.path.exists(config.PATH_DATA) == False:
            os.mkdir(config.PATH_DATA)

    async def evaluate_coco(self, file):
        
        fpath = f'{config.PATH_DATA}/{file.filename}'
        await save_json(file, fpath)

        metrics = metrics_ap.main_all(config.PATH_ANNOTATION, fpath, fmt = 'summarize')
        return metrics

    async def evaluate_coco_each(self, anns):

        # coco = COCO(fpath)
        # imgIds = coco.getImgIds()
        # annIds = coco.getAnnIds(imgIds=[imgIds[0]], iscrowd=False)
        # anns = coco.loadAnns(annIds)
        img_id = 0
        metrics = metrics_ap.main_each(config.PATH_ANNOTATION, img_id, anns)
        # print(metrics)
        return metrics

import os
import json
import shutil
import config
import metrics_ap
from pycocotools.coco import COCO
import uuid

async def save_json(file, path):
    with open(path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    

class cocoMetrics():
    def __init__(self):
        if os.path.exists(config.PATH_DATA) == False:
            os.mkdir(config.PATH_DATA)

    async def evaluate_coco(self, file):
        
        fpath = f'{config.PATH_DATA}/{str(uuid.uuid4())}.json'
        await save_json(file, fpath)

        metrics = metrics_ap.main_all(config.PATH_ANNOTATION, fpath, fmt = 'summarize')

        os.remove(fpath)
        return metrics

    async def evaluate_coco_each(self, img_id, anns):

        # coco = COCO(fpath)
        # imgIds = coco.getImgIds()
        # annIds = coco.getAnnIds(imgIds=[imgIds[0]], iscrowd=False)
        # anns = coco.loadAnns(annIds)
        metrics = metrics_ap.main_each(config.PATH_ANNOTATION, img_id, anns)
        # print(metrics)
        return metrics

    async def pycocotoolsEvaluation(self, file, annType='bbox'):
        # fpath = f'{config.PATH_DATA}/{file.filename}'
        fpath = f'{config.PATH_DATA}/{str(uuid.uuid4())}.json'
        await save_json(file, fpath)

        ret = metrics_ap.pycocotoolsEvaluation(config.PATH_ANNOTATION, fpath, annType)
        os.remove(fpath)
        return ret

    async def annsStatics(self, file, annType='bbox', n_anchors=9):

        import numpy as np
        from sklearn.cluster import KMeans

        # fpath = f'{config.PATH_DATA}/{file.filename}'
        fpath = f'{config.PATH_DATA}/{str(uuid.uuid4())}.json'
        await save_json(file, fpath)

        coco_json = json.load(open(fpath, 'r'))
        anns = coco_json['annotations']

        bbox_list = list()
        for ann_loop in anns:
            # print(ann_loop)
            bbox_list.append([ann_loop['bbox'][2], ann_loop['bbox'][3]])

        bbox_list = np.array(bbox_list)

        
        km = KMeans(n_clusters = n_anchors, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        y_km = km.fit(bbox_list)

        centroids = np.array(y_km.cluster_centers_)
        centroids_size = centroids[:, 0] * centroids[:, 1]
        arg_sort = np.argsort(centroids_size)
        centroids = centroids[arg_sort]

        os.remove(fpath)
        return centroids.tolist()





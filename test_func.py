
import copy
import json
import numpy as np
from pycocotools.coco import COCO

def create_testFile(path_gt, path_pred):

    coco_gt = COCO(path_gt)

    img_test = list()
    ann_test = list()
    imgIds = coco_gt.getImgIds()
    
    for id_img in imgIds:
        
        img = coco_gt.loadImgs([id_img])
        img_test += img
        # print(img_test)

        annIds_gt = coco_gt.getAnnIds(imgIds=[id_img], iscrowd=False)
        anns_gt = coco_gt.loadAnns(annIds_gt)

        for loop in anns_gt:
            temp = loop['bbox']
            temp[2] += np.random.randint(0, 20)
            temp[3] += np.random.randint(0, 20)
            data = {
                'id': loop['id'],
                'image_id': loop['image_id'],
                'bbox': temp, 
                'category_id': loop['category_id'], 
                'iscrowd': loop['iscrowd'], 
                'score': np.random.randint(5, 10) * 0.1 
            }
            ann_test += [data]
            print([data])

        # print(ann_test)


    # with open(path_gt, mode='r') as file_gt:
    #     test = json.load(file_gt)

    test = {'images':img_test,
            'annotations': ann_test
    }
    # test['annotations'] = ann_test
    # print(gt_json['images'])
    with open(path_pred, 'wt') as f:
        json.dump(test, f)


if __name__ == '__main__':

    import metrics_ap
    path_gt = './annotations/instances_val2017.json'
    path_pred = './annotations/test.json'


    # create_testFile(path_gt, path_pred)

    metrics_ap.main(path_gt, path_pred)

from pycocotools.coco import COCO
import numpy as np

def iou_1list(a, b_list):

    a_area = a[2] * a[3]
    b_area = b_list[:, 2] * b_list[:, 3]

    abx_mn = np.maximum(a[0], b_list[:, 0])
    aby_mn = np.maximum(a[1], b_list[:, 1])
    abx_mx = np.minimum(a[0] + a[2], b_list[:, 0] + b_list[:, 2])
    aby_mx = np.minimum(a[1] + a[3], b_list[:, 1] + b_list[:, 3])

    w = np.maximum(0, abx_mx - abx_mn)
    h = np.maximum(0, aby_mx - aby_mn)

    intersect = w * h

    iou = intersect / (a_area + b_area - intersect)
    return iou

def bbox_iou_loop(bbox1, bbox2):

    temp = list()
    for loop in bbox1:
        temp.append(iou_1list(loop, bbox2)[np.newaxis, :])

    return np.concatenate(temp, axis=0)

# def bbox_iou(bbox1, bbox2):
#     """
#     boxes1 (ndarray[N, 4]) – first set of boxes
#     boxes2 (ndarray[M, 4]) – second set of boxes
#     returns ndarray[N, M]
#     """
#     boxes1_area = bbox1[..., 2] * bbox1[..., 3]
#     boxes2_area = bbox2[..., 2] * bbox2[..., 3]
#     # boxes1 = np.concatenate([bbox1[..., :2] - bbox1[..., 2:4] * 0.5,  
#     #                          bbox1[..., :2] + bbox1[..., 2:4] * 0.5], axis= -1 \
#     # )
#     # boxes2 = np.concatenate([bbox2[..., :2] - bbox2[..., 2:4] * 0.5,  
#     #                          bbox2[..., :2] + bbox2[..., 2:4] * 0.5], axis= -1 \
#     # )
#     boxes1 = np.concatenate([bbox1[..., :2],  
#                              bbox1[..., :2] + bbox1[..., 2:4]], axis= -1 \
#     )
#     boxes2 = np.concatenate([bbox2[..., :2],  
#                              bbox2[..., :2] + bbox2[..., 2:4]], axis= -1 \
#     )

# def bbox_iou(bbox1, bbox2):
#     """
#     boxes1 (ndarray[N, 4]) – first set of boxes
#     boxes2 (ndarray[M, 4]) – second set of boxes
#     returns ndarray[N, M]
#     """
#     boxes1_area = bbox1[..., 2] * bbox1[..., 3]
#     boxes2_area = bbox2[..., 2] * bbox2[..., 3]
#     # boxes1 = np.concatenate([bbox1[..., :2] - bbox1[..., 2:4] * 0.5,  
#     #                          bbox1[..., :2] + bbox1[..., 2:4] * 0.5], axis= -1 \
#     # )
#     # boxes2 = np.concatenate([bbox2[..., :2] - bbox2[..., 2:4] * 0.5,  
#     #                          bbox2[..., :2] + bbox2[..., 2:4] * 0.5], axis= -1 \
#     # )
#     boxes1 = np.concatenate([bbox1[..., :2],  
#                              bbox1[..., :2] + bbox1[..., 2:4]], axis= -1 \
#     )
#     boxes2 = np.concatenate([bbox2[..., :2],  
#                              bbox2[..., :2] + bbox2[..., 2:4]], axis= -1 \
#     )

#     left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
#     right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
#     inter_section = np.maximum(right_down - left_up, 0.0)
#     inter_area = inter_section[..., 0] * inter_section[..., 1]
#     union_area = boxes1_area + boxes2_area - inter_area
#     IoU = inter_area / union_area
#     return IoU
#     left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
#     right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
#     inter_section = np.maximum(right_down - left_up, 0.0)
#     inter_area = inter_section[..., 0] * inter_section[..., 1]
#     union_area = boxes1_area + boxes2_area - inter_area
#     IoU = inter_area / union_area
#     return IoU


def convert_gt(anns, func_convert):

    if func_convert is None:
        ret = np.concatenate([ np.array(x["bbox"] + [x["category_id"]])[:, np.newaxis]  for x in anns], 1).T
    else:
        ret = np.concatenate([ np.array(x["bbox"] + [func_convert(x["category_id"])])[:, np.newaxis] for x in anns], 0).T
    return ret

def convert_pred(anns, func_convert):

    if func_convert is None:
        ret = np.concatenate([ np.array(x["bbox"] + [x["category_id"], x["score"]])[:, np.newaxis]  for x in anns], 1).T
    else:
        ret = np.concatenate([ np.array(x["bbox"] + [func_convert(x["category_id"]), x["score"]])[:, np.newaxis] for x in anns], 0).T
    return ret

def compute_each(anns_pred, anns_gt, func_convert = None):
    """
    bbox -> [top-left-x, top-left-y, width, height]
    category_id -> 

    "bbox": [473.07,395.93,38.65,28.67], 
    "category_id": 18
    "score"

    """

    iouv = np.linspace(0.5, 0.95, 10)
    if len(anns_gt) == 0:
        return (np.zeros((0, iouv.shape[0])), np.array([]), np.array([]), np.array([]))
    bbox_array_gt = convert_gt(anns_gt, func_convert)
    gt_category = bbox_array_gt[:, 4]

    if len(anns_pred) == 0:
        return (np.zeros((0, iouv.shape[0])), np.array([]), np.array([]), gt_category)
    
    bbox_array_pred = convert_pred(anns_pred, func_convert)
    pred_category = bbox_array_pred[:, 4]
    pred_score = bbox_array_pred[:, 5]
    
    correct = np.zeros((bbox_array_pred.shape[0], iouv.shape[0]), dtype=np.bool)
    
    for cls_loop in np.unique(pred_category):
        
        detected_list = list()
        pred_idx = np.where(bbox_array_pred[:, 4] == cls_loop)[0] # ti = torch.nonzero(_cls == tcls_tensor).view(-1)
        gt_idx = np.where(bbox_array_gt[:, 4] == cls_loop)[0] 

        if pred_idx.shape[0] == 0:
            continue
        
        # ious = bbox_iou(bbox_array_pred[pred_idx, :], bbox_array_gt[gt_idx, :]) # (N, M)
        ious = bbox_iou_loop(bbox_array_pred[pred_idx, :], bbox_array_gt[gt_idx, :]) # (N, M)
        
        ious_max_gt = np.max(ious, axis=1) #  (N)
        ious_max_gt_arg = np.argmax(ious, axis=1) # (N), the most fitted predicted bbox

        for index_pred, ious_max_gt_loop in enumerate(ious_max_gt):

            if ious_max_gt_loop < iouv[0]:
                continue
            
            detected = gt_idx[ious_max_gt_arg[index_pred]]

            if detected not in detected_list:
                detected_list.append(detected)
                
                correct[pred_idx[index_pred]] = (ious_max_gt[index_pred] > iouv)
                if len(detected_list) == bbox_array_gt.shape[0]:
                    break
    
    return (correct, pred_score, pred_category, gt_category)


def compute_ap(recall, precision):
    """
    """

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i-1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i+1] - mrec[i]) * mpre[i+1])
    return ap



def AveragePresicion(tp, score, pred_cls, target_cls):

    i = np.argsort(-score)
    tp, score, pred_cls = tp[i], score[i], pred_cls[i]

    pred_cls = pred_cls.astype(np.int32)
    target_cls = target_cls.astype(np.int32)

    unique_classes = np.unique(target_cls)
    pr_score = 0.1
    s = [unique_classes.shape[0], tp.shape[1]] # number of class, tp
    ap, p, r = np.zeros(s), np.zeros(s), np.zeros(s)

    for ci, c in enumerate(unique_classes):
        
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 or n_gt == 0:
            continue
        else:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall = tpc / (n_gt + 1e-12)
            r[ci] = round(np.interp(-pr_score, -score[i], recall[:, 0]), 3)

            precision = tpc / (tpc + fpc)
            p[ci] = round(np.interp(-pr_score, -score[i], precision[:, 0]), 3)

            for j in range(tp.shape[1]):
                ap[ci, j] = round(compute_ap(recall[:, j], precision[:, j]), 3)

    f1 = 2 * p * r / (p + r + 1e-12)
    return p, r, ap, f1, unique_classes.astype(np.int32)


def AveragePresicion_All(tp, score, pred_cls, target_cls):

    i = np.argsort(-score)
    tp, score, pred_cls = tp[i], score[i], pred_cls[i]

    pred_cls = pred_cls.astype(np.int32)
    target_cls = target_cls.astype(np.int32)

    unique_classes = np.unique(target_cls)
    
    ap, p, r = [], [], []

    for ci, c in enumerate(unique_classes):
        
        i = pred_cls == c
        n_gt = (target_cls == c).sum()
        n_p = i.sum()

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            p.append(0)
            r.append(0)
        else:
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            recall_curve = tpc / (n_gt + 1e-12)
            r.append(recall_curve[-1])

            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            ap.append(compute_ap(recall_curve, precision_curve))

    # print(ap)
    f1 = 2 * p * r / (p + r + 1e-12)
    return p, r, ap, f1, unique_classes.astype(np.int32)


def main_each(path_gt, id_img, anns_pred, func_convert = None, fmt='summarize'):

    coco_gt = COCO(path_gt)

    annIds_gt = coco_gt.getAnnIds(imgIds=[id_img], iscrowd=False)
    anns_gt = coco_gt.loadAnns(annIds_gt)

    temp = compute_each(anns_pred, anns_gt, func_convert=func_convert)

    stats = [np.concatenate(x, 0) for x in list(zip(*[temp]))]
    precision, recall, ap, f1, ap_class = AveragePresicion(*stats)
    AP, f1 = ap.mean(), ap[:, 0]

    if fmt == 'summarize':
        metrics = {
                'precision': precision[:, 0].mean(),
                'recall': recall[:, 0].mean(),
                'mAP': AP.tolist(),
                'f1': f1.tolist()
        }
    else:
        metrics = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'mAP': AP.tolist(),
            'f1': f1.tolist()
        }

    return metrics


def main_all(path_gt, path_pred, func_convert = None, fmt='summarize'):

    # print(path_pred, path_gt)
    coco_pred = COCO(path_pred)
    coco_gt = COCO(path_gt)

    imgIds = coco_gt.getImgIds()
    stats_list = list()
    for id_img in imgIds:

        annIds_gt = coco_gt.getAnnIds(imgIds=[id_img], iscrowd=False)
        annIds_pred = coco_pred.getAnnIds(imgIds=[id_img], iscrowd=False)
        anns_gt = coco_gt.loadAnns(annIds_gt)
        anns_pred = coco_pred.loadAnns(annIds_pred)

        temp = compute_each(anns_pred, anns_gt, func_convert=func_convert)
        stats_list.append(temp)

    stats = [np.concatenate(x, 0) for x in list(zip(*stats_list))]
    # precision, recall, ap, f1, ap_class = AveragePresicion_All(*stats)
    precision, recall, ap, f1, ap_class = AveragePresicion(*stats)
    

    AP, f1 = ap.mean(), ap[:, 0]

    if fmt == 'summarize':
        metrics = {
            'precision': precision[:, 0].mean(),
            'recall': recall[:, 0].mean(),
            'mAP': AP.tolist(),
            'f1': f1.tolist()
        }
    else:
        metrics = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'mAP': AP.tolist(),
            'f1': f1.tolist()
        }
    
    # print(precision.shape) # (80, 10)
    # print(recall.shape) # (80, 10)
    # print(ap.shape) # (80, 10)
    # print(f1.shape) # (80)
    # print(metrics)
    return metrics
    


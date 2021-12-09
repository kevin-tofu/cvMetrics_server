
## Server for Computer Vision Metrics  
This server evaluates the accuracy of the computer vision task on the COCO data set 
predicted by the machine learning model.

## API

| Route | Method | Query / Body | Description |
| --- | --- | --- | --- |
| /meanAveragePrecision | POST | - | Post a json file (written in COCO format) to get mean average precision. |
| /meanAveragePrecision_each | POST | - | Post a list of annotations (written in COCO format) to get mean average precision on an image.|
| /pycocotoolsEvaluation | POST | - | Post a json file (written in COCO format) to evaluate with pycocotools.|
| /bboxStatics | POST | - | Post a json file (written in COCO format) to get statics of bbox. This info is useful to make anchors for object detection model like YOLO.|


### Definition of annotations list for meanAveragePrecision_each-post API

```
{
    image_id: int
    annotation: [
        {
            iscrowd: int
            image_id: int
            bbox: list[float]
            category_id: int
            id: int
         },,,
    ]
}
```

## Environment variables
| Variable | required | Description |
| --- | --- | --- |
| APP_PORT | false | The port to which the application listens to, default is set to 80 |
| PATH_ANNOTATION | false | A PATH for annotation file |
| PATH_IMAGE | false | A PATH for directory that test-images are stored. |



## Server for Computer Vision Metrics  
This server evaluates the accuracy of the computer vision task on the COCO data set 
predicted by the machine learning model.

## API

| Route | Method | Query / Body | Description |
| --- | --- | --- | --- |
| /meanAveragePrecision | POST | - | Post a json file (written in COCO format) to get mean average precision. |
| /meanAveragePrecision_each | POST | - | Post a list of annotations (written in COCO format) to get mean average precision on an image.|

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
| PATH_DATA | false | Directry name where to store data. |
| PATH_ANNOTATION | false | A PATH for annotation file |
| PATH_IMAGE | false | A PATH for directory that val-images are stored. |


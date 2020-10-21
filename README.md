# Train detect

## How to use?

1) To infer on an image that is stored on your local machine
```
python yolo.py --image-path='/path/to/image/'
```
2) To infer on a video that is stored on your local machine
```
python yolo.py --video-path='/path/to/video/'
```

## Package install

`conda create -n XXX python=3.6`
`conda install numpy`
`conda install -c conda-forge opencv`
`pip install yolov3`


### To analysis the noise impact for residents besides MRT, this is the new way to get the time period when train passbys by detecting train from video.
#### 
- Using OpenCV to decomposite video to frames and get the images, timestamp from each frame.
- Using Yolo to detect train, if there is train from the image, then record the timestamp.
- Output the time period by applying the algorithm to calculate the train passbys' timestamps.


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

### 为了分析地铁声音对建筑的影响，从视频中检测出地铁，以及对应出现的时间段，从而可以快速拿到地铁通过时的音频。
#### 
- 使用opencv把视频分解成每个frame，获取每个frame对应的图片和时间点。
- 使用yolo预测图片上是否含有train，如果含有，获取对应的时间
- 把获取到的时间整理为时间段。并输出


## Overview
This project contains algorithms for small flying object detection with a **2-camera stereo rig**

## Code
`calibration.py` - functions for calibrating horizontal and vertical shift between the left and right frame

`sky_detection.py` - functions for thresholding and detecting sky region on the frame

`sky_detection_debug.py` - visualization of the sky detection algorithms for debugging and optimization

`main.py` - algorithm that detects objects **in the sky**, can be used for moving cameras

`detection.py` - algorithm that detects moving objects **when the camera is static**

`opencv_utils.py` - simple functions for visualizing debug info

`yolo.py` - simple code that runs YOLOv8n detection on any video. 


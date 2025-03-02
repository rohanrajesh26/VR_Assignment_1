# VR_Assignment_1

## Requirements

To run this project, you need the following libraries installed:

- OpenCV
- NumPy
- Matplotlib
- Pillow

You can install these libraries using pip:

```sh
pip install opencv-python-headless numpy matplotlib pillow
```

## How to Run

### 1. Coin Detection (1.py)

This script processes the image `old_coins.png` to detect and count the number of coins. It performs the following:

- Applies adaptive thresholding
- Uses Canny edge detection  
- Segments individual coins
- Displays results with coin count

To run:
```sh
python 1.py
```

Output:
- Displays interactive plots showing detection steps
- Creates `output_1.png` with detected coins
- Creates `output_segmented_coins.png` showing individual coins

### 2. Image Stitching (2.py) 

This script creates a panorama from three London Parliament images:
- `big-ben-clock _end.jpg`
- `big-ben-clock _mid.jpg`
- `big-ben-clock.jpg`

Features:
- SIFT keypoint detection
- Feature matching
- Image stitching
- Enhancement and cropping

To run:
```sh
python 2.py
```

Output:
- Displays keypoint matches between images
- Creates `final_panorama.jpg`

Note: Ensure all image files are in the same directory as the scripts.
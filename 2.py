#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

def load_images():
    """
    Load the three London Parliament images.
    Returns list of loaded images in OpenCV format.
    """
    image_names = ["/mnt/c/Users/rohan/OneDrive/Documents/VR_Assign1/big-ben-clock _end.jpg", "big-ben-clock _mid.jpg", "big-ben-clock.jpg"]
    images = []
    
    for img_name in image_names:
        # Read image
        img = cv2.imread(img_name)
        if img is None:
            raise ValueError(f"Could not read image: {img_name}")
        
        # Resize image to a manageable size while maintaining aspect ratio
        max_dimension = 1000
        height, width = img.shape[:2]
        scaling_factor = max_dimension / max(height, width)
        if scaling_factor < 1:
            img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
            
        images.append(img)
    
    return images

def detect_and_match_keypoints(images):
    """
    Detect and match keypoints between multiple images using SIFT.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Lists to store keypoints and descriptors
    keypoints_list = []
    descriptors_list = []
    
    # Detect keypoints in all images
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        keypoints_list.append(keypoints)
        descriptors_list.append(descriptors)
    
    return keypoints_list, descriptors_list

def match_images(descriptors1, descriptors2, ratio=0.75):
    """
    Match features between two images using ratio test.
    """
    # Initialize matcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    
    return good_matches

def visualize_results(images, keypoints_list, descriptors_list):
    """
    Visualize keypoints and matches between images.
    """
    # Visualize keypoints
    for i, (img, keypoints) in enumerate(zip(images, keypoints_list)):
        img_keypoints = cv2.drawKeypoints(
            img, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
        plt.title(f'Keypoints in Image {i+1}')
        plt.axis('off')
        plt.show()
    
    # Visualize matches between consecutive pairs
    for i in range(len(images)-1):
        matches = match_images(descriptors_list[i], descriptors_list[i+1])
        
        img_matches = cv2.drawMatches(
            images[i], keypoints_list[i],
            images[i+1], keypoints_list[i+1],
            matches[:50],  # Show top 50 matches for clarity
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
        )
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f'Matches between Image {i+1} and Image {i+2}')
        plt.axis('off')
        plt.show()

def stitch_images(images):
    """
    Stitch multiple images into a panorama using OpenCV's Stitcher
    """
    # Initialize OpenCV's stitcher
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    
    # Perform stitching
    status, panorama = stitcher.stitch(images)
    
    if status != cv2.Stitcher_OK:
        error_messages = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
        }
        error_msg = error_messages.get(status, f"Unknown error: {status}")
        raise Exception(f"Stitching failed: {error_msg}")
    
    return panorama

def enhance_and_crop_panorama(panorama):
    """
    Enhance the panorama and crop black borders
    """
    # Convert to float32 for processing
    panorama_float = panorama.astype(np.float32) / 255.0
    
    # Increase contrast and brightness
    contrast = 1.2
    brightness = 1.1
    panorama_enhanced = cv2.pow(panorama_float, contrast) * brightness
    
    # Clip values to valid range
    panorama_enhanced = np.clip(panorama_enhanced * 255, 0, 255).astype(np.uint8)
    
    # Convert to grayscale for border detection
    gray = cv2.cvtColor(panorama_enhanced, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours to crop black borders
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)
        cropped = panorama_enhanced[y:y+h, x:x+w]
        return cropped
    
    return panorama_enhanced

def analyze_keypoints(keypoints_list, matches_list):
    """
    Analyze and print statistics about detected keypoints and matches.
    """
    print("\nKeypoint Analysis:")
    for i, keypoints in enumerate(keypoints_list):
        print(f"Image {i+1}: {len(keypoints)} keypoints detected")
    
    print("\nMatch Analysis:")
    for i, matches in enumerate(matches_list):
        print(f"Between Image {i+1} and Image {i+2}: {len(matches)} good matches")

def main():
    """
    Main function to run both keypoint detection and image stitching
    """
    try:
        # Load images
        print("Loading images...")
        images = load_images()
        
        # Part A: Keypoint Detection
        print("\nPart A: Detecting and matching keypoints...")
        keypoints_list, descriptors_list = detect_and_match_keypoints(images)
        
        # Match between consecutive pairs
        matches_list = []
        for i in range(len(images)-1):
            matches = match_images(descriptors_list[i], descriptors_list[i+1])
            matches_list.append(matches)
        
        # Analyze keypoint results
        analyze_keypoints(keypoints_list, matches_list)
        
        # Visualize keypoint results
        print("\nGenerating keypoint visualizations...")
        visualize_results(images, keypoints_list, descriptors_list)
        
        # Part B: Image Stitching
        print("\nPart B: Creating panorama...")
        panorama = stitch_images(images)
        
        # Enhance and crop panorama
        print("Enhancing and cropping panorama...")
        final_panorama = enhance_and_crop_panorama(panorama)
        
        # Save panorama
        cv2.imwrite("final_panorama.jpg", final_panorama)
        
        # Display final panorama
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
        plt.title('Final Stitched Panorama')
        plt.axis('off')
        plt.show()
        
        print("\nProcessing completed successfully!")
        print("Final panorama saved as 'final_panorama.jpg'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()

# %%

"""
Face Verification Metrics Classification Script

This script performs face verification between ground truth images and predicted images
using different face recognition models from the DeepFace library.

Dependencies:
    - deepface
    - glob
    - natsort

Author: Matheus Souza
Date: 2025-03-20
"""

# Import required libraries
from deepface import DeepFace
import glob
from natsort import natsorted
from tqdm import tqdm

# Define available face recognition models
models = ["VGG-Face","Dlib"]

# Define paths for ground truth and predicted images
path_gt = "/home/medeirmv/Desktop/LSI_Paper/Datasets_LatentSpace/image_256x256/"
path_pred = "/home/medeirmv/Desktop/LSI_Paper/lsi/dmd_lsi/LatentSpaceImaging/verify/"

# Load and sort image paths
images_gt = sorted(glob.glob(path_gt+"*.jpg"))
images_pred = natsorted(glob.glob(path_pred+"*.png"))

fail_gt = {"Dlib": 50, "VGG-Face": 50}

# Iterate through each model and perform verification
for j in range(len(models)):
  # Initialize counters
  total = 0      # Counter for successful verifications
  fail = 0       # Counter for failed comparisons
  counter = 0    # Total number of comparisons attempted
  
  # Open log file for current model
  with tqdm(total=len(images_pred)) as pbar:
    with open("log_"+models[j]+".txt", "w") as log_file:
        # Process each pair of images
        for i in range(len(images_pred)):
            # Some images are not recognized as a face image by the model. They are skipped.
            try:
                # Perform face verification between ground truth and predicted image
                result = DeepFace.verify(
                    img1_path=images_gt[i],
                    img2_path=images_pred[i],
                    model_name=models[j]
                )
                
                # Log verification results
                log_file.write(f"Image: {images_gt[i].split('/')[-1]}, Verified: {result['verified']}, Distance: {result['distance']}\n")

                # Update successful verification counter
                if result['verified']:
                    total+=1
                    
            except Exception as e:
                # Handle and log failed comparisons
                fail += 1
                log_file.write(f"Fail, Image GT: {images_gt[i]}, Image Pred: {images_pred[i]}\n")

            counter += 1
            pbar.update(1)
            
        # Calculate and log final statistics
        if counter-fail != 0:
            print(counter)
            print(fail)
            print(total/(counter-fail))
            print(total/(len(images_pred)))
            log_file.write(f"counter: {total}, fail: {fail}, score: {total/(counter-fail)}, score_total: {total/(len(images_pred))},score_without_gt_fail: {total/(len(images_pred) - fail_gt[models[j]])}")
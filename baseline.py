import torch
import numpy as np
import argparse
import pickle
import cv2
import os
from sklearn.linear_model import LinearRegression
from PIL import Image
from clip_interrogator import Config, Interrogator
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--annotations", type=str, required=True, help="Annotations Pickle File")
parser.add_argument("--video_root", type=str, required=True, help="Folder containing video files")

args = parser.parse_args()

assert os.path.exists(args.annotations)#Check for annotations file
annotation_file = open(args.annotations, 'rb')
annotations = pickle.load(annotation_file)#Load annotations
annotation_file.close()

for video in annotations.keys():#Check that videos exist
    assert os.path.exists(os.path.join(args.video_root, video+".mp4"))
    
#Open results file
results_file = open("results.csv", 'w')
results_file.write("ID,Driver_State_Changed")
for i in range(23):
    results_file.write(f",Hazard_Track_{i},Hazard_Name_{i}")
results_file.write("\n")
#Setup captioning model
ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

for video in sorted(list(annotations.keys())):#Iterate through available videos
    
    video_stream = cv2.VideoCapture(os.path.join(args.video_root, video+".mp4"))
    assert video_stream.isOpened()
    
    frame = 0
    previous_centroids = []
    median_dists = []
    captioned_tracks = {}
    driver_state_flag = False
    while video_stream.isOpened():
        print(f'{video}_{frame}')
        ret, frame_image = video_stream.read()
        if ret == False: #False means end of video or error
            assert frame == len(annotations[video].keys()) #End of the video must be final frame
            break
            
        #Gather BBoxes from annotations
        bboxes = []
        centroids = []
        chips = []
        track_ids = []
        for ann_type in ['challenge_object']:
            for i in range(len(annotations[video][frame][ann_type])):
                x1, y1, x2, y2 = annotations[video][frame][ann_type][i]['bbox']
                track_ids.append(annotations[video][frame][ann_type][i]['track_id'])
                bboxes.append([x1, y1, x2, y2])
                centroids.append([x1+(abs(x2-x1)/2),y1+(abs(y2-y1)/2)])
                chips.append(frame_image[int(y1):int(y2), int(x1):int(x2)])
        bboxes = np.array(bboxes)
        centroids = np.array(centroids)
        
        if len(bboxes) == 0 or len(previous_centroids) == 0:
            frame +=1
            if len(centroids) !=0:
                previous_centroids.append(centroids)
            continue #We can't make a prediction of state change w/o knowing the previous state
        
        
        ###Driver state change detection
        dists = []
        for centroid in centroids:
            potential_dists = np.linalg.norm(previous_centroids - centroid, axis=1)
            min_dist = np.sort(potential_dists)[0]
            dists.append(min_dist)
            
        median_dist = np.median(dists)#Take the median to reduce noise
        
        median_dists.append(median_dist)
        
        #We are using median dist as a proxy for speed
        #If we have no prior measurements, we can't tell if we're slowing down
        if len(median_dists) == 1:
            frame +=1
            continue
            
        
        
        x = np.array(range(len(median_dists))).reshape(-1, 1)
        y = np.array(median_dists)
        speed_model = LinearRegression().fit(x, y)
    
        if speed_model.coef_[0] < 0: #If we are slowing down, driver state has probably changed
            driver_state_flag = True
            
        
        ###Detect the hazard
        #The hazard may be in the center of the screen (ie in front of the car)
        #This works for late detections, but for objects that jump in front,
        #More sophisticated solutions might be needed
        
        image_center = [frame_image.shape[1]/2, frame_image.shape[0]/2]
        potential_hazard_dists = np.linalg.norm(centroids-image_center, axis=1)
        probable_hazard = np.argmin(potential_hazard_dists)
        hazard_track = track_ids[probable_hazard]
        
        ###Hazard description
        #Simple captioning model for hazard chip
        #DO NOT INCLUDE COMMAs (,) or special characters in description
        if hazard_track not in captioned_tracks:
            hazard_chip = cv2.cvtColor(chips[probable_hazard], cv2.COLOR_BGR2RGB)
            hazard_chip = Image.fromarray(hazard_chip)
            # Generate caption
            hazard_caption = ci.interrogate(hazard_chip)

            hazard_caption = hazard_caption.replace(","," ")
            
            captioned_tracks[hazard_track] = hazard_caption
        else:
            hazard_caption = captioned_tracks[hazard_track]#Why compute caption again? Re-use hazard desc
            
        results_file.write(f"{video}_{frame},{driver_state_flag},{hazard_track},{hazard_caption}" + "".join([", , " for i in range(22)])+'\n')
        
        frame +=1
        
results_file.close()



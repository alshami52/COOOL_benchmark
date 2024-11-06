# Challenge Of Out-Of-Label (COOOL) Benchmark for Hazard Detection
The COOOL benchmark is a collection of dashcam oriented videos (currently ~200) which have been annotated by human labelers to identify objects of interest and potential hazards to the vehicle driving.
The dataset includes a wide variety of hazards and neussiance objects.
Due to the size of the dataset and complexity of the data, this is an evaluation only benchmark.

# WACV 2025 Challenge
The WACV2025 COOOL workshop challenge comprises of three goals fundamental to hazard detection and avoidance on the road:
- Determine if the driver has reacted to a potential hazard (slowing down, avoiding, or deciding to ignore)
- Determine what annotation is hazardous (by bounding box)
- Caption the hazard

While performance on these tasks will be evaluated together, they are not necessarily co-occuring. For example, some of our annotations actually determine what the hazard is before the driver in the video reacted, so a driver-state change will not necessarily accompany all hazards.

# Scoring
CSV Format:
- All submissions must be in the form of a results.csv file
- Due to variable abounts of hazards in each video/frame, please include 22 Hazard_Track and Hazard_Name columns for each csv row
- Predicting extra (non-present) hazards or names will be counted against you, as will incorrect predictions
- Hazard_Track positions should be an integer.
- Predicted Hazard Tracks and Names go together in pairs, ie Hazard_Name_1 must contain a caption for Hazard_Track_1.
- Captions for hazards must not contain commas (,) due to the submission file format
- Driver State should only change once per-video (we are only considering the driver's initial reaction), 'False' means no state change from the beginning of the video, 'True' means the state has changed.

Scoring Rules:
- Each frame has an annotation for 'Driver_State_Change', score will be correct_state_change_prediction/total_frames
- The number of present hazards is variable among videos, score will be correctly_predicted_hazards/max(known_hazards, len(predicted_hazards)) PER FRAME
- The number of present hazards is variable among videos, score will be correctly_predicted_hazard_names/max(known_hazard_names, len(predicted_hazards_names)) PER FRAME

Scores will be macro averaged to produce a final result for ranking.

A sample results file is present in this repo for your convenience, please adhere to the formatting or your submission may not be scored.

# General Rules
- Solutions must be automated, no humans in-the-loop
- Any publicly available dataset is allowed for pre-training, and any model (not trained on the COOOL Benchmark) is allowed for use in competition solutions
- We encourage the use of Captioning, LLM (including those which require an API), Open-set, Open-Vocabulary, or Object Detection models.
- Do NOT stuff random words or large-dictionaries into the caption field, we will check for this.
- While your code is your own and you are not required to release it as part of this competition, all results must be reproducible (within reason).
- If we suspect cheating, we will request to review your solution code, if we detect a rule violation or you refuse to share the code which produced your results.csv, we will remove you from the challenge.
- These rules are subject to change

# Baseline
Download the linked videos into a folder on your machine/server, install the required packages from 'baseline.py' and run 'baseline.py --video_root [path_to_video_folder] --annotations [path_to_annotations_pkl]'
The necessary pkl files are present in this repo.
The given baseline satasfies the three tasks using straightforward methods. Using logistic regression to detect when bounding boxes start moving slowly, we determine driver_state_change. By determining the closest bounding box to the center of the image, we predict the hazardous object. We use clip interrogator to generate a caption for that object.


Significant improvements can be made using advanced models for captioning, speed or car orientation detection, or even hazard trajectory prediction.

# Google Doc Videos 
https://docs.google.com/document/d/1XjYL2zFI3JxAscmP0uLQPQsAi3jBte5Yih64kVa0B7U/edit?usp=sharing

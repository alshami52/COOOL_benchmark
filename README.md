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
- Due to variable amounts of hazards in each video/frame, please include 22 Hazard_Track and Hazard_Name columns for each CSV row
- Predicting extra (non-present) hazards or names will be counted against you, as well as incorrect predictions
- Hazard_Track positions should be an integer.
- Predicted Hazard Tracks and Names go together in pairs, i.e., Hazard_Name_1 must contain a caption for Hazard_Track_1.
- Captions for hazards must not contain commas (,) due to the submission file format
- Driver State should only change once per video (we are only considering the driver's initial reaction), 'False' means no state change from the beginning of the video, and 'True' means the state has changed.

Scoring Rules:
- Each frame has an annotation for 'Driver_State_Change'; score will be correct_state_change_prediction/total_frames
- The number of present hazards is variable among videos; score will be correctly_predicted_hazards/max(known_hazards, len(predicted_hazards)) PER FRAME
- The number of present hazards is variable among videos; score will be correctly_predicted_hazard_names/max(known_hazard_names, len(predicted_hazards_names)) PER FRAME

Scores will be macro-averaged to produce a final ranking result.

A sample results file is present in this repo for your convenience; please adhere to the formatting, or your submission may not be scored.

# General Rules
- Solutions must be automated, with no humans in-the-loop
- Any publicly available dataset is allowed for pre-training, and any model (not trained on the COOOL Benchmark) is allowed for use in competition solutions
- We encourage using Captioning, LLM (including those that require an API), Open-set, Open-Vocabulary, or Object Detection models.
- Do NOT stuff random words or large-dictionaries into the caption field, we will check for this, only a max of 35 characters are allowed. If you submit captions longer than 35 characters, we will only score the first 35.
- All results must be reproducible (within reason).
- If we suspect cheating, we will request to review your solution code; if we detect a rule violation or you refuse to share the code that produced your results.csv, we will remove you from the challenge.
- These rules are subject to change

# Baseline
Download the linked videos into a folder on your machine/server, install the required packages from 'baseline.py' and run 'baseline.py --video_root [path_to_video_folder] --annotations [path_to_annotations_pkl]'
The necessary pkl files are present in this repo.
The given baseline satisfies the three tasks using straightforward methods. Using logistic regression to detect when bounding boxes start moving slowly, we determine driver_state_change. By determining the closest bounding box to the center of the image, we predict the hazardous object. We use a clip interrogator to generate a caption for that object.


Significant improvements can be made using advanced models for captioning, speed or car orientation detection, or even hazard trajectory prediction.

# Google Doc Videos 
[Video Folder](https://drive.google.com/drive/folders/1u7MtzXH2kmZEAvEhQgHkS1p0VYLmzCw2)
#Citation
If you find our work useful, please cite it using the following BibTeX entry:
@misc{alshami2024cooolchallengeoutoflabelnovel,
      title={COOOL: Challenge Of Out-Of-Label A Novel Benchmark for Autonomous Driving}, 
      author={Ali K. AlShami and Ananya Kalita and Ryan Rabinowitz and Khang Lam and Rishabh Bezbarua and Terrance Boult and Jugal Kalita},
      year={2024},
      eprint={2412.05462},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.05462}, 
}
# License
This project is released under an [MIT License](https://github.com/alshami52/COOOL_benchmark/blob/main/LICENSE).

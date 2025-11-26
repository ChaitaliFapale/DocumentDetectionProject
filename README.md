# Document_Detection_Project

Document Boundary Detection Using DETR + OpenCV
1. Project Overview
This project focuses on detecting the boundaries of identity card-like documents—such as passports—using a combination of DETR (DEtection TRansformer) and OpenCV.
The model can identify the presence of a passport even under rotation, skew, noise, or partial occlusion.
OpenCV post-processing is used to refine boundaries.

2. Key Features
Uses DETR for object detection with high accuracy.
Handles skewed, rotated, and partially occluded passport images using OpenCV geometric processing.
Supports data augmentation for improved robustness.
Simple end-to-end pipeline: prepare data → train DETR → detect boundaries.


3. Requirements
Python 3.10+
PyTorch 2.0+
HuggingFace Transformers
OpenCV
NumPy
Matplotlib
Install dependencies:

pip install -r requirements.txt

4. Setup
Create & activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

Install packages
pip install --upgrade pip
pip install -r requirements.txt

5. Data Preparation
Place original document/passport images inside:
datasets/images/

Ensure annotations are available in:
datasets/annotations.xml


Convert XML annotations to DETR/COCO-style JSON:
python datasets/convert_annotations.py
This generates detection labels suitable for DETR training.


7. Training the DETR Model
python src/train_detr.py

This script:
Loads COCO-formatted dataset
Fine-tunes facebook/detr-resnet-50
Saves the model in:
models/detr_passport/

8. Inference
Run detection on new images:
python src/infer_detr.py --source test.png

Pipeline:
DETR predicts bounding boxes for passports.
OpenCV refines:
Document boundary detection

Output:
Visualized results with detected boundaries.


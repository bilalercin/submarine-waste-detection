# Submarine Waste Detection

This project aims to train and evaluate a YOLOv8-based object detection model for detecting waste in underwater images.

## Table of Contents
- [Project Purpose](#project-purpose)
- [Background](#background)
- [Technologies and Libraries Used](#technologies-and-libraries-used)
- [Dataset Details](#dataset-details)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Sample Results & Visualization](#sample-results--visualization)
- [Contribution and License](#contribution-and-license)

## Project Purpose
Automatic detection of waste in underwater environments is critical for monitoring and cleaning marine pollution. In this project, a deep learning model is trained on a dataset containing various types of waste and marine life.

## Background
Marine pollution is a growing environmental problem, with waste materials accumulating in oceans and seas. Detecting and classifying underwater waste using computer vision and artificial intelligence can help automate monitoring and support clean-up operations. This project leverages state-of-the-art object detection techniques to address this challenge.

## Technologies and Libraries Used
- Python 3.x
- Jupyter Notebook
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV
- Numpy, Pandas
- Scikit-learn
- Matplotlib
- PyYAML
- tqdm

## Dataset Details
The project uses a dataset of underwater images labeled in COCO format. The dataset includes:
- **Images:** Underwater scenes containing both marine life and various types of waste.
- **Annotations:** COCO-style JSON files with bounding boxes and class labels.
- **Classes:** 22 different categories, including marine animals (fish, starfish, crab, etc.) and waste types (bottle, bag, net, rope, etc.).

## Data Preparation
1. **COCO to YOLO Conversion:**
   - The provided script converts COCO JSON annotations to YOLO format, which is required for training with YOLOv8.
2. **Directory Structure:**
   - Images and labels are organized into `train` and `val` folders for training and validation.
3. **Data Configuration:**
   - A `data.yaml` file is generated, specifying paths and class names for YOLO training.

## Model Architecture
- **YOLOv8:**
  - A state-of-the-art, real-time object detection architecture.
  - Pre-trained weights (`yolov8n.pt`) are used as a starting point.
  - The model is fine-tuned on the underwater waste dataset.

## Training Process
- **Parameters:**
  - Epochs: 50
  - Image size: 640x640
  - Batch size: 16
  - Optimizer: Default (Adam or SGD)
- **Steps:**
  1. Install dependencies.
  2. Prepare data and configuration files.
  3. Train the model using the Ultralytics YOLO CLI or Python API.

## Evaluation Metrics
- **Precision & Recall:** Measure the accuracy of waste detection and classification.
- **mAP (mean Average Precision):** Evaluates detection performance across all classes.
- **Class-wise Reports:** Detailed metrics for each class (e.g., bottle, bag, fish, etc.).
- **Confusion Matrix:** Visualizes model performance and misclassifications.

## Usage
1. **Install the required libraries:**
   ```bash
   pip install ultralytics opencv-python numpy pandas scikit-learn matplotlib pyyaml tqdm
   ```
2. **Open the notebook file and run the steps in order:**
   - `ai_final03.ipynb`
3. **Train the model:**
   - Follow the notebook cells to preprocess data, train, and evaluate the model.
4. **Evaluate and visualize results:**
   - The notebook includes code for generating evaluation metrics and visualizations.

## Sample Results & Visualization
- The model can successfully detect 22 different classes in underwater images.
- Training and validation results, including precision, recall, mAP, and confusion matrices, are reported in detail within the notebook.
- Example visualizations:
  - Bounding boxes on detected objects
  - Class-wise performance tables
  - Confusion matrix plots

## Contribution and License
- **Educational Use:** This project is for educational and research purposes.
- **How to Contribute:**
  - Fork the repository
  - Create a new branch for your feature or bugfix
  - Submit a pull request with a clear description of your changes
- **License:**
  - Unless otherwise stated, this project is open for academic and non-commercial use. Please cite the repository if you use it in your research.

---

For more information, please check the notebook file or contact the repository owner.

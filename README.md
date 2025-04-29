# 👁️ Iris Recognition System

A simple iris recognition system using Python, OpenCV, Gabor filters, and cosine similarity. The system segments the iris from two uploaded eye images, normalizes them, extracts features using Gabor filters, and compares them.

## 📌 Features

- Eye image preprocessing
- Iris segmentation using Hough Circle Transform
- Iris normalization
- Feature extraction using Gabor filters
- Matching using cosine similarity
- Visualization of each processing step

## 🛠️ Requirements

Install required packages:
```bash
pip install opencv-python-headless scipy scikit-image scikit-learn matplotlib
```

## 🚀 How It Works
Upload two eye images.

Preprocess and segment the iris region.

Normalize the segmented iris.

Apply Gabor filter for feature extraction.

Compare features using cosine similarity.

Display similarity score and visual steps.

## 📊 Output Example
Similarity Score: 0.9123

Match Result: ✅ Match Found

Includes side-by-side visualizations:

Original Eye

Segmented Iris

Normalized Iris

Gabor-enhanced Iris

## 📁 File Structure
```bash
Copy
Edit
iris-recognition/
│
├── iris_match.py         # Main script
├── iris_match.ipynb      # (Optional) Jupyter Notebook version
├── README.md             # Project description
└── requirements.txt      # Package dependencies
```
🖼️ Sample Visualization

## 📸 Notes
Use clear, front-facing eye images.

If iris can't be detected, try higher quality or better-lit images.

## 📌 To-Do
Improve iris segmentation using Daugman’s integro-differential operator.

Add eye image dataset loader.

Deploy as a web app (e.g., with Streamlit).

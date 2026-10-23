# Facial Emotion Recognition (FER) System Using CNN

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework-Flask](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning-TensorFlow/Keras](https://img.shields.io/badge/Deep%20Learning-TensorFlow%20%2F%20Keras-orange.svg)](https://tensorflow.org/)
[![Computer Vision-OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-red.svg)](https://opencv.org/)
[![License-MIT](https://img.shields.io/badge/Academic-Minor%20Project-lightgrey.svg)]()

### College Minor Project — 5th Semester
- **Department**: Department of Computer Science and Engineering — Artificial Intelligence & Machine Learning (CSE - AI&ML)
- **Institution**: Lords Institute of Engineering and Technology (UGC Autonomous), Hyderabad
- **Academic Year**: 2025–2026
- **Student**: **SYED MOAZAM**
- **Roll Number**: **160923748131**
- **Internal Guide**: **Ms. Zeba Ruqshanh** *(Assistant Professor)*

---

## 📌 Project Overview
Facial Emotion Recognition (FER) is an affective computing and computer vision application that automatically classifies human facial expressions into distinct emotions. This project implements an end-to-end deep learning pipeline using Convolutional Neural Networks (CNNs) integrated into a Flask web application, allowing users to upload facial images and obtain instant emotion predictions.

The system is trained and evaluated on the benchmark **FER-2013** dataset across seven basic human affective states:
1. **Happy**
2. **Sad**
3. **Angry**
4. **Fear**
5. **Surprise**
6. **Neutral**
7. **Disgust**

---

## 🏗️ System Architecture & Workflow

```
┌─────────────────┐       ┌───────────────────────┐       ┌────────────────────────┐
│  Input Image    │ ───>  │  Image Preprocessing  │ ───>  │    Trained CNN Model   │
│  (JPG / PNG)    │       │  • Grayscale Convert  │       │  (`emotiondetector.h5`)│
└─────────────────┘       │  • Face ROI Detection │       │  • Conv2D + BatchNorm  │
                          │  • Resize to 64x64    │       │  • MaxPooling + Dropout│
                          │  • [0, 1] Normalize   │       │  • Dense + Softmax     │
                          └───────────────────────┘       └────────────────────────┘
                                                                       │
                                                                       ▼
                                                          ┌────────────────────────┐
                                                          │   Predicted Emotion    │
                                                          │   e.g. "happy" (97.3%) │
                                                          │   Rendered on Flask UI │
                                                          └────────────────────────┘
```

---

## 📂 Project Structure

```text
├── docs/
│   ├── FER_Minor_Project_Report_5th_Sem.pdf         # Minor Final Project Report
│   ├── FER_Minor_Project_Report_5th_Sem.docx        # Word Source Document
│   ├── FER_Minor_Project_Presentation_5th_Sem.pdf   # Seminar Presentation Slides
│   └── FER_Minor_Project_Presentation_5th_Sem.pptx  # PowerPoint Source Presentation
├── images/
│   └── test/
│       └── happy/        # Test samples from report (4 (1).jpg to 4 (10).jpg)
├── static/
│   └── samples/          # Test sample images for web interface
├── templates/
│   └── result.html       # Clean Flask web UI matching Report Ch 8 & Ch 9
├── uploads/              # Temporary upload storage (with .gitkeep)
├── app.py                # Flask backend & model inference server
├── emotiondetector.h5    # Trained CNN weights file (64x64 grayscale input)
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

---

## 📊 Dataset & Sample Images
The test images are taken from the benchmark **FER-2013** dataset (hosted on [Kaggle](https://www.kaggle.com/c/fer2013)):
* **Directory**: `images/test/happy/`
* **Primary Sample**: `4 (1).jpg` achieves **97.3%** confidence for the `happy` class.
* **Additional Samples**: `4 (2).jpg` through `4 (10).jpg` matching Figure 9.2 in the Final Project Report.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/mrdf-13/FER.git
cd FER
```

### 2. Create and Activate a Virtual Environment
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Web Application
```bash
python app.py
```
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## 🖥️ User Interface & Output
The user interface is designed strictly in accordance with **Chapter 8 (Section 8.1.2)** and **Chapter 9 (Figures 9.1–9.4)** of the Minor Project Report:
* **Figure 9.1**: User View with dashed image upload container and predict button.
* **Figure 9.2**: Selection of test images from `images/test/happy/`.
* **Figure 9.3**: Image preprocessing and input submission.
* **Figure 9.4**: Display of the processed face and output: `Predicted Expression: happy`.

---

## 📚 Documentation & Deliverables
* **Project Report (PDF)**: [`docs/FER_Minor_Project_Report_5th_Sem.pdf`](docs/FER_Minor_Project_Report_5th_Sem.pdf)
* **Project Report (Word)**: [`docs/FER_Minor_Project_Report_5th_Sem.docx`](docs/FER_Minor_Project_Report_5th_Sem.docx)
* **Presentation Slides (PDF)**: [`docs/FER_Minor_Project_Presentation_5th_Sem.pdf`](docs/FER_Minor_Project_Presentation_5th_Sem.pdf)
* **Presentation Slides (PPTX)**: [`docs/FER_Minor_Project_Presentation_5th_Sem.pptx`](docs/FER_Minor_Project_Presentation_5th_Sem.pptx)

---

## 👤 Author & Attribution
- **Student**: Syed Moazam (Roll No: `160923748131`)
- **Institution**: Lords Institute of Engineering and Technology, Hyderabad
- **Department**: Computer Science and Engineering (AI&ML)
- **Supervision**: Ms. Zeba Ruqshanh, Assistant Professor, Dept. of CSE (AI&ML)

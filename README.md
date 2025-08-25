# Deepfake Detector using Hybrid CNN and LSTM Models 🕵️‍♂️

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9%2B-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)  
[![Keras](https://img.shields.io/badge/Keras-2.9%2B-red.svg?logo=keras&logoColor=white)](https://keras.io/)  
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg?logo=opencv&logoColor=white)](https://opencv.org/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-1.10%2B-ff69b4.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)  
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)  

A Deepfake Detection application built using a combination of image processing and deep learning techniques. This project leverages a hybrid CNN and LSTM model to identify manipulated videos.

---

## 📁 Repository Structure

```

deepfake-detector/
├── .gitignore
├── README.md
├── app.py
└── requirements.txt

````

---

## 📄 Project Overview

This project presents a **Deepfake Detector application** that analyzes video files to determine if they contain synthetic, AI-generated content (deepfakes).  
The system is designed as a **user-friendly tool** for detecting video manipulation.

---

## ⚙️ Methodology

The core of the detection system is a **hybrid deep learning model**. The process involves:

1. **Video Processing**: Using OpenCV, the application extracts individual frames from the input video.  
2. **Face Detection**: For each frame, a face is detected and isolated.  
3. **Feature Extraction**: An **InceptionV3** model (pre-trained on a large dataset) is used to extract spatial features from detected faces.  
4. **Temporal Analysis**: Features from a sequence of frames are fed into an **RNN** that analyzes temporal inconsistencies/artifacts common in deepfakes.  
5. **Prediction**: The model outputs a probability score indicating the likelihood of the video being a deepfake, which is then displayed to the user.

---

## 🛠️ Technologies Used

- **Python** – Core programming language  
- **TensorFlow & Keras** – Deep learning framework  
- **OpenCV** – Video and image manipulation (frame extraction, face detection)  
- **NumPy & Pandas** – Data handling and numerical operations  
- **Streamlit** – Interactive web-based UI  

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/deepfake-detector.git
cd deepfake-detector
````

Create a virtual environment (recommended):

```bash
python -m venv venv
# Activate it:
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application with:

```bash
streamlit run app.py
```

This will:

* Launch a local web server
* Open the app in your default browser
* Let you upload a video file for deepfake detection

---

## 📦 Requirements

`requirements.txt` includes:

```
tensorflow>=2.9.0
keras>=2.9.0
opencv-python>=4.5.5.64
streamlit>=1.10.0
numpy>=1.22.4
pandas>=1.4.3
```

---

## ⚠️ Disclaimer

> This application uses a **demonstration model** and is not fully trained for production use.
> For critical analysis, a robust, fully trained deepfake detection system should be employed.

---

## 👨‍🏫 Credits

Project based on the Final Year Project Report from the **Department of Computer Engineering, at the University of Lahore**.
Supervised by **Dr. M. Wasim Nawaz**.

---


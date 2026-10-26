# 🔫 🔪 Gun & Knife Detection (YOLO11n)

This repository contains a real-time dangerous object detection application developed using the modern **YOLO11n** architecture. The system is specifically trained to detect weapons such as **guns** and **knives** to enhance security screening and monitoring capabilities.

## 🚀 Live Demo
You can try the live interactive web application directly on Hugging Face Spaces:
👉 [Click here to view the Live Demo](https://huggingface.co)

## 🛠️ Tech Stack & Architecture
* **Model:** YOLO11n (Ultralytics)
* **Framework:** Streamlit (For the web interface)
* **Deployment:** Docker containerized environment
* **Base Model weights:** Loaded from `cosgun99/gun-knife-yolo11n`

## 📦 How to Run Locally

If you want to run this project on your local machine, follow these simple steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd gun-knife-detection-yolo11n
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app:**
   ```bash
   streamlit run app.py
   ```

## ⚖️ License
Distributed under the **MIT License**. See `LICENSE` for more information.

# Cure Better 

**Cure Better** is an AI-powered facial expression analysis tool designed to estimate **pain levels** in real time using a standard webcam.  
It leverages [MediaPipe FaceMesh](https://developers.google.com/mediapipe/solutions/vision/face_mesh), OpenCV, and custom feature extraction heuristics to calculate a **Pain Score (0–100)** and classify it into bands:  
- **Mild**
- **Moderate**
- **High**

Snapshots are automatically saved when high pain levels are detected.

---

##  Features
-  **Real-time webcam tracking** using MediaPipe FaceMesh  
-  **Pain score estimation** from brows, eyes, mouth, and forehead landmarks  
-  **Baseline calibration** – press `b` to capture your neutral face  
-  **Automatic snapshots** on high pain detection  
-  **Lightweight** – CPU-only, no GPU required  

---

##  Installation

1. **Clone the repo**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/cure-better.git
   cd cure-better
   
2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   
 4.  **Install dependencies**:
    
    ```bash
   pip install -r requirements.txt

5. **Usage**:
   Run the app:
   ```bash
   python cure_better.py
\
   **Controls**:

  Press b → capture baseline (neutral face, no smile/frown)
  
  Press ESC → quit
  
  Snapshots of high pain moments are stored in the snapshots/ folder

6. **Output**:

  On-screen overlay shows:
  
  Pain Score (0–100)
  
  Band classification (None / Mild / Moderate / High)

  Debug dots drawn for forehead, brows, eyes, mouth, and cheeks

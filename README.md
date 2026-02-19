# 🦺 AI Safety Monitoring System (PPE Detection)

An advanced **AI-powered Safety Monitoring Dashboard** built with **Streamlit** and the **Roboflow API** for detecting Personal Protective Equipment (PPE) compliance in real-time.

This project helps monitor workplace safety by detecting:
- ✅ Helmets  
- ✅ Safety Vests / Jackets  
- ❌ Missing PPE (Unsafe workers)

It supports:
- Image Upload Analysis  
- Live Webcam Feed Detection  
- Video Upload Frame-by-Frame Monitoring  
- Email Alerts for Safety Violations  

---

## 🚀 Features

### 🖼️ Image Upload PPE Detection
- Upload any image (`jpg`, `png`, `jpeg`)
- Detect PPE compliance
- Download annotated images
- View safety compliance chart and individual details

### 🎥 Live Webcam Monitoring
- Real-time PPE detection from webcam
- Adjustable detection intervals
- Dashboard metrics:
  - Total persons detected
  - Safe workers
  - Unsafe workers
- Automatic email alerts on violations

### 🎞️ Video Upload Analysis
- Upload videos (`mp4`, `avi`, `mkv`, `mov`)
- Frame-by-frame PPE detection
- Live progress updates with result table
- Generate downloadable annotated video output

---

## 📊 Dashboard Preview

The dashboard provides:
- Bounding box visualization
- Compliance pie charts
- Individual safety reports

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **OpenCV**
- **Roboflow Detection API**
- **Matplotlib**
- **Pandas**
- **SMTP Email Alerts**

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-safety-monitoring.git
cd ai-safety-monitoring
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate    # For Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 Roboflow Setup

1. Create an account at **Roboflow**
2. Train or use a PPE detection model
3. Copy your:
   - API Key
   - Model Endpoint (example: `ppe-detection/1`)

Enter them in the sidebar of the Streamlit app.

---

## 📧 Email Alert Setup (Optional)

Enable email alerts in sidebar:
- Sender Email (Gmail recommended)
- Gmail App Password
- Receiver Email

⚠️ Make sure you use a Gmail **App Password**, not your normal password.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## 📂 Project Structure

```
📁 AI-Safety-Monitoring/
│── app.py                # Main Streamlit dashboard
│── requirements.txt      # Dependencies
│── README.md             # Documentation
```

---

## ⚠️ Safety Use Case

This system can be deployed in:
- Construction sites  
- Industrial workplaces  
- Manufacturing plants  
- Hazardous zones  

Ensuring workers wear proper PPE reduces accidents and improves safety compliance.

---

## ✨ Future Enhancements
- Multi-camera monitoring
- SMS/WhatsApp alerts
- PPE compliance report export
- Cloud deployment support

---

## 🤝 Contributing
Pull requests are welcome!  
For major changes, please open an issue first to discuss improvements.

---

## 📜 License
This project is for educational and research purposes.

---

### 👨‍💻 Developed By
**Your Name**  
AI & Data Science Student  


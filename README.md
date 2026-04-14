# 🧠 AI-Powered Smart Iron with Fabric Classification

An intelligent IoT-based system that automatically classifies fabric types and recommends optimal ironing temperature using Machine Learning and real-time sensor data.

---

## 🚀 Overview

This project integrates **IoT + AI/ML** to build a smart ironing system.  
It collects real-time data (temperature, motion, static time) using sensors and uses a **Voting Ensemble Machine Learning model** to classify fabrics such as Cotton, Silk, Wool, Polyester, and detect anomalies.

---

## 🎯 Key Features

- 🔍 Real-time fabric classification using ML
- 🌡️ Intelligent temperature recommendation
- 📡 IoT-based data collection using ESP8266
- ☁️ Cloud integration with ThingSpeak
- 🌐 Web dashboard for monitoring
- ⚡ High accuracy using ensemble learning

---

## 🛠️ Tech Stack

### Hardware
- ESP8266 (NodeMCU)
- LM35 Temperature Sensor
- MPU6050 (Accelerometer + Gyroscope)

### Software
- Python
- Flask
- Scikit-learn
- HTML, CSS, JavaScript
- ThingSpeak Cloud

---

## 🧠 Machine Learning Model

- **Model Type:** Voting Ensemble (Soft Voting)
- **Algorithms Used:**
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - Support Vector Machine (SVM)
- **Techniques:**
  - Feature Engineering
  - Cross Validation (Stratified K-Fold)
  - Model Optimization

---

## 🔄 System Workflow

1. Sensor data collection (Temperature, Motion, Static Time)
2. Data preprocessing & feature extraction
3. ML model prediction
4. Fabric classification output
5. Temperature recommendation
6. Display on web dashboard & cloud upload

---

## 📊 Dataset

- Custom dataset generated using real-time sensor readings
- Features:
  - Temperature_C
  - Motion_Variation
  - Static_Time_s
- Target:
  - Fabric_Type (Cotton, Silk, Wool, Polyester, Anomaly)

---

## 📁 Project Structure

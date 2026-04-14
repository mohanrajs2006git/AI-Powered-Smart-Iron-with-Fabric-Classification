# AI-Powered Smart Iron with Fabric Classification

## Overview
This project presents an intelligent IoT-based smart ironing system that integrates machine learning to automatically classify fabric types and recommend optimal temperature settings. The system leverages real-time sensor data and a robust ensemble learning model to enhance safety, efficiency, and user convenience.

---

## Key Features
- Real-time fabric classification using machine learning  
- Intelligent temperature recommendation based on fabric type  
- IoT-based data acquisition using ESP8266 (NodeMCU)  
- Cloud integration for monitoring and data visualization  
- Web-based dashboard for real-time insights  
- High accuracy using Voting Ensemble model  

---

## Technology Stack

### Hardware
- ESP8266 (NodeMCU)  
- LM35 Temperature Sensor  
- MPU6050 (Accelerometer & Gyroscope)  

### Software
- Python  
- Flask (Backend API)  
- Scikit-learn (Machine Learning)  
- HTML, CSS, JavaScript (Frontend)  
- ThingSpeak (Cloud Platform)  

---

## Machine Learning Model
The system utilizes a **Voting Ensemble (Soft Voting Classifier)** combining multiple algorithms to achieve high accuracy and reliability.

**Algorithms Used:**
- Random Forest  
- Extra Trees  
- Gradient Boosting  
- Support Vector Machine (SVM)  

**Techniques Applied:**
- Feature Engineering (e.g., Heat Exposure Index, Ratios, Log Transformations)  
- Stratified Cross-Validation  
- Model Evaluation using Accuracy, Precision, Recall, and F1-score  

---

## System Workflow
1. Sensor data acquisition (Temperature, Motion, Static Time)  
2. Data preprocessing and feature extraction  
3. Machine learning model prediction  
4. Fabric classification output generation  
5. Temperature recommendation  
6. Data visualization on web dashboard and cloud  

---


## Project Structure

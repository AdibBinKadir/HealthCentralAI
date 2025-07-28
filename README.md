# 🧠 Health Central AI – Your Personal First Aid & Mental Health Chatbot

**Health Central AI** is an intelligent, always-available chatbot designed to guide users through first aid emergencies and offer mental health support using a clinically-informed, AI-powered backend.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Why Health Central AI?](#why-health-central-ai)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Demo](#demo)
- [Contributors](#contributors)

---

## 📖 About the Project

**Health Central AI** is built to deliver **accurate, evidence-based responses** during critical moments — such as medical emergencies or mental health challenges.

The platform empowers users with:
- Instant **first aid instructions**
- Grounded **mental health support**
- Conversational **general wellness guidance**

---

## ❓ Why Health Central AI?

Emergencies and emotional crises don’t wait — and neither should help.

- 🩹 Most people are unaware of **proper first aid procedures**.
- 🧘 Many suffer from **anxiety and stress** without support.
- 💬 Existing chatbots often lack **domain-specific intelligence**.

**Health Central AI** bridges that gap with an AI model trained on **doctor-annotated medical data**, ensuring reliable help is always a message away.

---

## ⚙️ How It Works

We implemented a **two-tiered response system**:

### 🧬 1. Custom AI Model (First Aid & Mental Health)
- Built using **TensorFlow** and **Keras**
- Trained on **doctor-annotated data**
- Utilizes a **neural network-based architecture**
- Handles queries related to **first aid scenarios and mental health support**

### 🤖 2. Gemini API (General Queries)
- Handles **non-medical or casual conversations**
- Ensures a seamless, intelligent chat experience even beyond our model’s scope

---

## 🧰 Tech Stack

| Layer        | Tools & Libraries                                  |
|--------------|----------------------------------------------------|
| **Backend**  | Flask, Python, TensorFlow, Keras                   |
| **Frontend** | HTML5, CSS3, Jinja2 templating                     |
| **Modeling** | NumPy, Pickle, JSON                                |
| **API**      | Gemini (for general conversation)                  |

---

## ✨ Key Features

- ✅ Doctor-trained deep learning model for first aid & mental health
- 💬 Seamless chatbot UI with Flask + Jinja2
- 🧘 Calm and supportive tone for mental wellness interactions
- 🌐 Two-tiered intelligence (custom AI + Gemini)
- 📦 Lightweight, fast, and easy to deploy

---

## 🚀 Setup & Installation

> ⚠️ Prerequisite: Python 3.10 or higher recommended

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/health-central-ai.git
cd health-central-ai
```
### 2. Create and Activate a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

- **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

- **On Windows**
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

  ```bash
  pip install -r requirements.txt
  ```

### 4. Set up Environment Variables

You will need a Google Gemini API key for the chatbot's fallback mechanism to work.

 1. Create a new file named keys.env in the root of the project directory.
 2. Add your API key to this file as follows:

  ```keys.env
  API_KEY="your_google_gemini_api_key_here"
  ```

  🔐 Note: The keys.env file is listed in .gitignore to ensure your secret keys are not accidentally committed to the repository.

---

## ▶️ Usage

There are two main ways to interact with this project: running the web application or retraining the model.

### Running the Web Application

To start the Flask server and interact with the chatbot in your browser:

 1. Make sure your virtual environment is activated.
 2. Run the run.py script:

  ```bash
  python run.py
  ```

  3. Open your web browser and navigate to: http://127.0.0.1:5000




### 🧠 Retraining the Model

 If you modify the `data/data.json` file with new intents or responses, you will need to retrain the model.

  1. Make sure your virtual environment is activated.
  2. Run the training.py script:

   ```bash
   python training.py
   ```

  This will automatically process the new data and overwrite the old model files in the models/ directory with the updated versions.

---

## 🤝 Contributors

This project was a collaborative effort. Big thanks to the entire team:

- **[Adib Bin Kadir](https://github.com/AdibBinKadir)** - Project Lead | Full-Stack Architecture, AI/ML & Deployment
- **[Athoiba Thongum](https://github.com/athoi-ba)** - Frontend Development & UI/UX Design
- **[Avirup Bhattacharjee](https://github.com/Avirup-Bhattacharjee)** - Data Collection & UI/UX Design





# 📉 Customer Churn Prediction App (Deployed on Render)

This project is an end-to-end machine learning web app for predicting customer churn using a trained classification model.

🔗 **Live App:** [https://churn-prediction-app-dxft.onrender.com](https://churn-prediction-app-dxft.onrender.com)

---

## 📸 App Preview

![Churn Prediction App Screenshot](screenshots/streamlit_ui.png)

---

## 🛠 Project Overview

This app:
- Trains a `RandomForestClassifier` to predict churn
- Encodes/preprocesses input features using `ColumnTransformer`
- Uses `joblib` to save/load model artifacts
- Provides a user-friendly interface using `Streamlit`
- Is deployed serverlessly using **Render**

---

## 📁 Project Structure

```
churn-prediction-app/
├── app/
│   └── app.py                  # Streamlit web interface
├── model/
│   ├── train_model.py          # Training logic
│   ├── preprocess.py           # Data loading/preprocessing
│   ├── churn_model.pkl         # Trained classifier
│   └── preprocessor.pkl        # Encoded transformer
├── data/
│   └── telco_churn.csv         # Simulated dataset
├── requirements.txt            # Python dependencies
├── render.yaml                 # Render deployment config
├── screenshots/
│   └── streamlit_ui.png        # App screenshot
└── README.md                   # Project documentation
```

---

## ⚙️ Local Setup

### ✅ 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-app.git
cd churn-prediction-app
```

### ✅ 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 3. Train the Model (optional if .pkl files exist)

```bash
python model/train_model.py
```

This saves:
- `model/churn_model.pkl`
- `model/preprocessor.pkl`

### ✅ 4. Launch the Streamlit App

```bash
streamlit run app/app.py
```

---

## 🌐 Deploying to Render

Render is a free serverless platform that supports Python + Streamlit.

### ✅ Setup Steps

1. Push all files to a GitHub repo
2. Go to [https://render.com](https://render.com)
3. Click **"New Web Service"**
4. Connect your GitHub repo
5. Configure the following:
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app/app.py --server.port $PORT`
6. Done! 🎉 Your app is live.

📌 Add a `render.yaml` like this to automate config:

```yaml
services:
  - type: web
    name: churn-prediction-app
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app/app.py --server.port $PORT
    autoDeploy: true
```

---

## 🤖 Tech Stack

| Purpose         | Tool            |
|----------------|-----------------|
| Language        | Python 3        |
| ML Library      | scikit-learn    |
| Web UI          | Streamlit       |
| Deployment      | Render          |
| Model Storage   | joblib          |
| Dataset         | Simulated Telco Churn |

---

## 📬 Author

**Paulo Cavallo**  
🔗 [LinkedIn](https://www.linkedin.com/in/paulo-cavallo/)  
🧠 [GitHub](https://github.com/YOUR_USERNAME)

---

## 📄 License

This project is available under the MIT License.
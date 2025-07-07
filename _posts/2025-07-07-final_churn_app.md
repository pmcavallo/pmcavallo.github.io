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
---

## 🧪 Model Training and Preprocessing (Python)

### 1. Load and Preprocess Data

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_and_split(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

    categorical = X.select_dtypes(include='object').columns.tolist()
    numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
    ])

    X_transformed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, preprocessor
```

---

### 2. Train and Save the Model

```python
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model(X_train, y_train, preprocessor):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'model/churn_model.pkl')
    joblib.dump(preprocessor, 'model/preprocessor.pkl')

    return model
```

---

### 3. Execute Training Script

```python
if __name__ == "__main__":
    df = load_data("data/telco_churn.csv")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df)
    train_and_save_model(X_train, y_train, preprocessor)
```

---

## 🧾 Notes

- We use `ColumnTransformer` to preprocess numerical and categorical features.
- The target `Churn` is binary encoded.
- Final model and preprocessor are saved as `.pkl` files for use in the web app.


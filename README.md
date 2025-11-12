import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# -------------------------------
# Authentication System
# -------------------------------
users = {"admin": hashlib.sha256("password".encode()).hexdigest()}

def check_login(username, password):
    return users.get(username) == hashlib.sha256(password.encode()).hexdigest()

# -------------------------------
# Load Dataset (No Upload)
# -------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\VICTUS\OneDrive\Desktop\Job Salary Prediction\Salary dataset.csv")
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
        if 'Race' in df.columns:
            df.drop(columns=['Race'], inplace=True)
        df.dropna(inplace=True)
        st.success("✅ Dataset loaded successfully!")
        return df
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        st.stop()

# -------------------------------
# Preprocessing & Model
# -------------------------------
def preprocess_data(df):
    label_encoders = {}
    categorical_columns = ['Gender', 'Education Level', 'Job Title', 'Country']
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            label_encoders[col] = le

    scaler = StandardScaler()
    if 'Years of Experience' in df.columns:
        df[['Years of Experience']] = scaler.fit_transform(df[['Years of Experience']])
    return df, label_encoders, scaler

def train_model(df):
    X = df[['Years of Experience', 'Education Level_encoded', 'Job Title_encoded']]
    y = df['Salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------------------
# Streamlit App Setup
# -------------------------------
st.set_page_config(page_title="💼 Job Salary Prediction", layout="wide")

# -------------------------------
# Centered Login Page
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("""
    <style>
    .centered-box {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 90vh;
    }
    .login-card {
        background-color: #fff5f7;
        padding: 40px 60px;
        border-radius: 20px;
        box-shadow: 0px 4px 30px rgba(255, 192, 203, 0.5);
        text-align: center;
        width: 400px;
        border: 1px solid #ffc0cb;
    }
    .login-card h2 {
        color: #c2185b;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="centered-box"><div class="login-card">', unsafe_allow_html=True)
    st.markdown("<h2>🔒 Login to Continue</h2>", unsafe_allow_html=True)

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_login(username, password):
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# -------------------------------
# Main Page
# -------------------------------
st.title("💼 Job Salary Prediction System")
st.markdown("#### Predict employee salary based on experience, education, job title, and country!")

df = load_data()
df, label_encoders, scaler = preprocess_data(df)
model = train_model(df)

# -------------------------------
# Sidebar Input
# -------------------------------
st.sidebar.header("🧾 Enter Details")
years_exp = st.sidebar.slider("Years of Experience", 0, 40, 5)
gender = st.sidebar.selectbox("Gender", df['Gender'].unique())
education = st.sidebar.selectbox("Education Level", df['Education Level'].unique())
job_title = st.sidebar.selectbox("Job Title", df['Job Title'].unique())

# ✅ Always include India
countries = df['Country'].unique().tolist()
if "India" not in countries:
    countries.append("India")
country = st.sidebar.selectbox("Country", countries)

currency_prefix_mapping = {
    "India": "₹", "United States": "$", "United Kingdom": "£",
    "Canada": "C$", "Australia": "A$", "Germany": "€", "France": "€",
    "Japan": "¥", "China": "¥", "Brazil": "R$", "South Africa": "R"
}
currency_prefix = currency_prefix_mapping.get(country, "$")

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("🎯 Predict Salary"):
    input_data = np.array([[years_exp]])
    input_data[:, 0] = scaler.transform(input_data[:, 0].reshape(-1, 1)).flatten()
    education_encoded = label_encoders['Education Level'].transform([education])[0]
    job_title_encoded = label_encoders['Job Title'].transform([job_title])[0]
    input_data = np.hstack([input_data, [[education_encoded, job_title_encoded]]])

    salary_pred = model.predict(input_data)[0]
    base_experience = 5
    if years_exp > base_experience:
        salary_pred *= (1 + 0.02 * (years_exp - base_experience))
    elif years_exp < base_experience:
        salary_pred *= (1 - 0.015 * (base_experience - years_exp))

    st.success("✅ Salary adjusted based on experience!")

    st.markdown(
        f"""
        <div style="
            background-color: #f0f9ff;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(100, 181, 246, 0.4);
            text-align: center;
            border: 1px solid #bbdefb;">
            <h2 style="color: #1976d2;">💰 Predicted Salary</h2>
            <h1 style="color: #1565c0; font-size: 50px;">{currency_prefix}{salary_pred:,.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    report = pd.DataFrame([[years_exp, gender, education, job_title, country, f"{currency_prefix}{salary_pred:,.2f}"]],
                          columns=['Years of Experience', 'Gender', 'Education Level', 'Job Title', 'Country', 'Predicted Salary'])
    csv = report.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Prediction Report", data=csv, file_name="salary_prediction.csv", mime="text/csv")

# -------------------------------
# Charts (Updated Colors)
# -------------------------------
st.subheader("📊 Salary Distribution")
fig1 = px.histogram(
    df, x="Salary", nbins=50, title="Salary Distribution",
    color_discrete_sequence=["#64b5f6"]
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📈 Salary vs Experience")
fig2 = px.scatter(
    df, x="Years of Experience", y="Salary", color="Education Level",
    title="Experience vs Salary", color_discrete_sequence=px.colors.sequential.Blues
)
st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #fff5f7;
}
[data-testid="stSidebar"] {
    background: #ffe4ec;
    color: #4a4a4a;
    border-right: 2px solid #ffc0cb;
}
h1, h2, h3, h4 {
    color: #1565c0;
    font-family: 'Segoe UI', sans-serif;
}
.stButton>button {
    background: linear-gradient(90deg, #64b5f6, #42a5f5);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #42a5f5, #64b5f6);
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

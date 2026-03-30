import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Prediction Tool", layout="centered")

st.title("🎯 Big Small Color Prediction Tool")

# ✅ Default dataset (no CSV needed)
data = {
    "number": [2,7,5,1,8,3,6,9,4,0,7,2,5,8,1,3,6],
    "color": ["red","green","violet","red","green","red","violet","green","red","violet","green","red","violet","green","red","red","violet"]
}

df = pd.DataFrame(data)

# ✅ Preprocess
df['size'] = df['number'].apply(lambda x: 1 if x >= 5 else 0)
color_map = {'red': 0, 'green': 1, 'violet': 2}
df['color_encoded'] = df['color'].map(color_map)

# ✅ Train models
X = df[['number']]

size_model = RandomForestClassifier()
size_model.fit(X, df['size'])

color_model = RandomForestClassifier()
color_model.fit(X, df['color_encoded'])

# 🎮 User Input
last_number = st.number_input("Enter Last Number (0-9)", min_value=0, max_value=9)

if st.button("Predict"):
    size_pred = size_model.predict([[last_number]])[0]
    color_pred = color_model.predict([[last_number]])[0]

    size = "Big" if size_pred == 1 else "Small"
    color_map_rev = {0: 'Red', 1: 'Green', 2: 'Violet'}
    color = color_map_rev[color_pred]

    st.success(f"Prediction: {size} | Color: {color}")

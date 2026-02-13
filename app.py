import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Set up the page title
st.set_page_config(page_title="Swiggy Recommender", layout="wide")
st.title("🍴 Swiggy Restaurant Recommendation System")
st.markdown("Discover the best places to eat based on your preferences!")

# --- LOAD DATA ---
@st.cache_data # This makes the app faster
def load_data():
    df_clean = pd.read_csv('cleaned_data.csv')
    df_encoded = pd.read_csv('encoded_data.csv')
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    return df_clean, df_encoded, encoder

try:
    df_clean, df_encoded, encoder = load_data()

    # --- SIDEBAR INPUTS ---
    st.sidebar.header("Filter your Preferences")
    
    # Dropdowns for City and Cuisine
    selected_city = st.sidebar.selectbox("Select City", sorted(df_clean['city'].unique()))
    selected_cuisine = st.sidebar.selectbox("Select Cuisine", sorted(df_clean['cuisine'].unique()))
    
    # Inputs for Rating and Cost
    min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)
    max_budget = st.sidebar.number_input("Max Budget (Cost for two)", value=500, step=50)

    # --- RECOMMENDATION LOGIC ---
    if st.sidebar.button("Show Recommendations"):
        # 1. Prepare User Input for the Encoder
        # We only take the first word of the cuisine to match our training data
        user_cuisine_fixed = selected_cuisine.split(',')[0].strip()
        user_input_cat = pd.DataFrame([[selected_city, user_cuisine_fixed]], columns=['city', 'cuisine'])
        
        # 2. Transform categorical input
        user_encoded = encoder.transform(user_input_cat)
        
        # 3. Combine with numerical input (Rating, 0 for count, Cost)
        user_num = np.array([[min_rating, 0, max_budget]])
        user_vector = np.hstack([user_num, user_encoded])
        
        # 4. Calculate Similarity
        similarities = cosine_similarity(user_vector, df_encoded)
        top_indices = similarities[0].argsort()[-5:][::-1]
        
        # 5. Display Results
        results = df_clean.iloc[top_indices]
        
        st.write(f"### Top 5 Recommendations in {selected_city}:")
        # Display nicely in a table
        st.dataframe(results[['name', 'city', 'cuisine', 'rating', 'cost', 'address']], use_container_width=True)
    else:
        st.info("Choose your preferences in the sidebar and click 'Show Recommendations'")

except FileNotFoundError:
    st.error("Error: 'cleaned_data.csv' or 'encoder.pkl' not found. Run your Jupyter Notebook steps first!")
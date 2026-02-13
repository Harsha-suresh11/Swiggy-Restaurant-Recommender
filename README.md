# Swiggy Restaurant Recommendation System 🍴

A Machine Learning-based recommendation engine that suggests the best restaurants based on user preferences like City, Cuisine, and Budget.

## 🚀 Features
* **Personalized Recommendations:** Uses Cosine Similarity to find the best matches.
* **Interactive UI:** Built with Streamlit for a seamless user experience.
* **Data-Driven:** Analyzes Swiggy's dataset to provide real-time suggestions.

## 🛠️ How I Built It
1. **Data Cleaning:** Removed "Too Few Ratings," handled missing values in the cost column, and converted text data to numerical formats.
2. **Preprocessing:** Used **One-Hot Encoding** to transform categorical data (City & Cuisine) into a mathematical format.
3. **Logic:** Implemented a **Cosine Similarity** matrix to compare user input against thousands of restaurant profiles.
4. **Deployment:** Created an interactive dashboard where users can filter by rating and budget.

## 📦 Requirements
To run this project locally, you need:
* Python 3.x
* Pandas
* Scikit-Learn
* Streamlit

## 💻 How to Run
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`# 

# CodeAlpha_Car-Price-Prediction-with-Machine-Learning
CodeAlpha_Car Price Prediction with Machine Learning also A webapp using streamlit

# Car Price Prediction App

A machine learning web app to predict the selling price of used cars, built with Streamlit and scikit-learn.

## Features
- Predicts car prices based on user input (brand, year, price, kms driven, fuel type, etc.)
- Interactive web interface using Streamlit
- Trained on a dataset of 300+ used cars
- Model options: Linear Regression, Random Forest, ElasticNet
- Visualizes prediction vs. input price

## Demo
![App Screenshot](https://images.unsplash.com/photo-1502877338535-766e1452684a)

## Files
- `app.py` — Streamlit app for car price prediction
- `car data.csv` — Dataset of used cars
- `car_price_pipeline_model.joblib` — Trained ML pipeline
- `code_alpha_car_price.ipynb` — Jupyter notebook (EDA, feature engineering, model training)
- `codealpha.mp4` — (Optional) Demo video

## How to Run
1. **Install requirements**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib requests
   ```
2. **Run the app**
   ```bash
   streamlit run app.py
   ```
3. **Open in browser**
   - Go to the local URL shown in the terminal (usually http://localhost:8501)

## Usage
- Enter car details in the sidebar and main form
- Click **Predict Price** to get the estimated selling price
- View a comparison bar chart of present price vs. predicted price

## Model Details
- Feature engineering includes brand extraction, car age, log of kms driven, and price per age
- Models compared: ElasticNet, RandomForest, LinearRegression
- Best model selected by cross-validation RMSE

## Author
[Muhammad Umar Farooqi](https://github.com/umarii04)

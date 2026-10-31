# 🚗 Car Resale Value Predictor

## 📌 Project Overview

The **Car Resale Value Predictor** is a Machine Learning application developed using Python and Streamlit to estimate the resale value of used cars based on various vehicle features.

The project uses the **K-Nearest Neighbors (KNN) Regression** algorithm to predict car prices based on features such as brand, location, manufacturing year, kilometers driven, fuel type, transmission, owner type, mileage, engine capacity, maximum power, and number of seats.

The application provides an interactive interface where users can enter car details and obtain an estimated resale value.

## ✨ Features

- Predicts the estimated resale value of used cars.
- Accepts multiple vehicle specifications as input.
- Encodes categorical features using Label Encoding.
- Applies feature scaling using MinMaxScaler.
- Uses KNN Regression for price prediction.
- Interactive web interface built with Streamlit.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- KNN Regression
- Streamlit
- Pickle

## ⚙️ Machine Learning Workflow

1. Data loading and exploration.
2. Data cleaning and preprocessing.
3. Encoding categorical variables using LabelEncoder.
4. Splitting the dataset into training and testing sets.
5. Feature scaling using MinMaxScaler.
6. Training the KNN Regression model.
7. Evaluating model performance.
8. Saving the trained model and preprocessing objects.
9. Deploying the model using Streamlit.

## 📊 Input Features

The model uses the following features:

- Brand
- Location
- Manufacturing Year
- Kilometers Driven
- Fuel Type
- Transmission
- Owner Type
- Mileage
- Engine Capacity
- Maximum Power
- Number of Seats

## 🚀 Installation and Execution

Install the required libraries:

```bash
pip install streamlit pandas numpy scikit-learn
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

- Data preprocessing and feature encoding.
- Feature scaling and transformation.
- Implementing KNN Regression.
- Evaluating regression model performance.
- Saving and loading ML models using Pickle.
- Developing interactive ML applications using Streamlit.

## 👨‍💻 Author

Developed as a hands-on Machine Learning project to explore regression algorithms and their real-world applications.

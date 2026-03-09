 import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score

def clean_data(df):
    """
    Cleans the cardekho dataset for model training.
    """
    df = df.replace(' ', np.nan)
    
    # Clean 'mileage(km/ltr/kg)'
    if 'mileage(km/ltr/kg)' in df.columns:
        df['mileage(km/ltr/kg)'] = df['mileage(km/ltr/kg)'].astype(str).str.split().str[0]
        df['mileage(km/ltr/kg)'] = pd.to_numeric(df['mileage(km/ltr/kg)'], errors='coerce')
        
    # Clean 'engine'
    if 'engine' in df.columns:
        df['engine'] = df['engine'].astype(str).str.split().str[0]
        df['engine'] = pd.to_numeric(df['engine'], errors='coerce')
        
    # Clean 'max_power'
    if 'max_power' in df.columns:
        # Some max_power values might be 'bhp' strings, handle gracefully
        df['max_power'] = df['max_power'].astype(str).str.split().str[0]
        # Handle cases where value is '' or 'null'
        df['max_power'] = df['max_power'].replace(['', 'null'], np.nan)
        df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')
        
    df = df.dropna()
    return df

def train_multiple_linear_regression(df):
    print("Training Multiple Linear Regression Model...")
    X = df[['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']]
    Y = df['selling_price']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    score = r2_score(Y_test, Y_pred)
    print(f"MLR R2 Score: {score:.4f}")
    
    # Save the model
    joblib.dump(model, 'mlr_model.pkl')
    print("Saved 'mlr_model.pkl'")

def train_simple_linear_regression(df):
    print("Training Simple Linear Regression Model...")
    X = df[['km_driven']]
    Y = df['selling_price']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    score = r2_score(Y_test, Y_pred)
    print(f"SLR R2 Score: {score:.4f}")
    
    # Save the model
    joblib.dump(model, 'slr_model.pkl')
    print("Saved 'slr_model.pkl'")

def train_polynomial_regression(df):
    print("Training Polynomial Regression Model...")
    X = df[['km_driven']]
    Y = df['selling_price']
    
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    score = r2_score(Y_test, Y_pred)
    print(f"Polynomial Regression R2 Score: {score:.4f}")
    
    # Save model and transformer
    joblib.dump(model, 'pr_model.pkl')
    joblib.dump(poly, 'poly_converter.pkl')
    print("Saved 'pr_model.pkl' and 'poly_converter.pkl'")

def train_logistic_regression(df):
    print("Training Logistic Regression Model...")
    
    # Create a binary classification target (e.g., above or below median price)
    median_price = df['selling_price'].median()
    print(f"Median Price used for classification: {median_price}")
    
    Y = (df['selling_price'] > median_price).astype(int)
    X = df[['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    print(f"Logistic Regression Accuracy: {acc:.4f}")
    
    # Save the model
    joblib.dump(model, 'log_reg_model.pkl')
    # Save the threshold so the app knows what "1" means
    joblib.dump(median_price, 'log_reg_threshold.pkl')
    print("Saved 'log_reg_model.pkl' and 'log_reg_threshold.pkl'")

def train_knn_classification(df):
    print("Training K-Nearest Neighbors Classification Model...")
    
    # Create binary classification target (above or below median price)
    median_price = df['selling_price'].median()
    print(f"KNN Classification - Median Price threshold: {median_price}")
    
    Y = (df['selling_price'] > median_price).astype(int)
    X = df[['year', 'km_driven', 'mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']]
    
    # KNN requires feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, Y_train)
    
    Y_pred = model.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    print(f"KNN Classification Accuracy: {acc:.4f}")
    
    # Save the model, scaler, and threshold
    joblib.dump(model, 'knn_clf_model.pkl')
    joblib.dump(scaler, 'knn_clf_scaler.pkl')
    joblib.dump(median_price, 'knn_clf_threshold.pkl')
    print("Saved 'knn_clf_model.pkl', 'knn_clf_scaler.pkl', and 'knn_clf_threshold.pkl'")

if __name__ == "__main__":
    try:
        # Load the dataset
        print("Loading cardekho.csv...")
        data = pd.read_csv("cardekho.csv")
        
        # Clean the data
        print("Cleaning data...")
        cleaned_data = clean_data(data)
        
        # Train MLR
        train_multiple_linear_regression(cleaned_data)
        
        print("-" * 30)
        
        # Train SLR
        train_simple_linear_regression(cleaned_data)
        
        print("-" * 30)
        
        # Train PR
        train_polynomial_regression(cleaned_data)
        
        print("-" * 30)
        
        # Train Logistic Regression
        train_logistic_regression(cleaned_data)
        
        print("-" * 30)
        
        # Train KNN Classifier
        train_knn_classification(cleaned_data)
        
        print("All models trained and saved successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_compare():
    # 1. Load the data
    df = pd.read_csv("data/processed/featured_border_data.csv")
    
    # 2. Data Cleaning: Drop those first-row NaNs we discussed
    df = df.dropna()
    
    # 3. Select our "Behavioral Clues" (Features)
    features = ['dist_moved', 'dist_to_border', 'angle_change']
    X = df[features]
    y = df['label']
    
    # 4. Split into Training (80%) and Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Train Model A: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    # 6. Train Model B: Logistic Regression (Baseline)
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    
    # 7. Evaluate both
    models = {"Random Forest": rf_model, "Logistic Regression": lr_model}
    
    for name, model in models.items():
        preds = model.predict(X_test)
        print(f"\n--- {name} Results ---")
        print(f"Accuracy: {accuracy_score(y_test, preds):.2f}")
        print(classification_report(y_test, preds))
    
    # 8. Save the best model
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf_model, "models/border_intruder_model.pkl")
    print("\n Best model saved to models/border_intruder_model.pkl")

if __name__ == "__main__":
    train_and_compare()
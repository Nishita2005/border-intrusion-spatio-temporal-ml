import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_elite_system():
    # 1. Load Model and Data
    pipeline = joblib.load('models/border_intruder_model.pkl')
    df = pd.read_csv('data/raw/border_data.csv')
    
    X = df.drop(['label', 'timestamp'], axis=1)
    y = df['label']

    print("--- 🛡️ Military Grade Validation ---")
    
    # 2. K-Fold Cross Validation (The Stress Test)
    scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean Reliability: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    # 3. Confusion Matrix (The 'False Alarm' Check)
    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Intruder'], 
                yticklabels=['Normal', 'Intruder'])
    plt.title('Confusion Matrix: Detection Accuracy')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('visuals/confusion_matrix.png')
    print("✅ Validation Complete. Matrix saved to visuals/ folder.")

if __name__ == "__main__":
    os.makedirs('visuals', exist_ok=True)
    evaluate_elite_system()
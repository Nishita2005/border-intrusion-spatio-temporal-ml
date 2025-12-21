import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def train_elite_model():
    # 1. Load the new data
    df = pd.read_csv('data/raw/border_data.csv')
    
    # 2. Define our features
    numeric_features = ['speed', 'angle_change', 'sensor_confidence']
    categorical_features = ['object_type', 'terrain', 'visibility']
    
    X = df[numeric_features + categorical_features]
    y = df['label']

    # 3. Create a "Preprocessor" (Handles numbers and text automatically)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # 4. Create the "Elite Pipeline"
    # This bundles the preprocessing and the AI model into one file!
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    # 6. Save the entire Pipeline
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/border_intruder_model.pkl')
    print(f"Step 2 Complete: Elite Model trained (Accuracy: {pipeline.score(X_test, y_test):.2f})")

if __name__ == "__main__":
    train_elite_model()
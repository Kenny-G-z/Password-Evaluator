import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

csv = "passwords.csv"
file = "passwords.pkl"
trees = 3000  

def train_and_save_model():
    if not os.path.exists(csv):
        raise FileNotFoundError(f"Error: '{csv}' not found.")

    df = pd.read_csv(csv, on_bad_lines="skip")
    
    pwd_col = "password" if "password" in df.columns else df.columns[0]
    label_col = "strength" if "strength" in df.columns else df.columns[1]
    
    df = df.dropna(subset=[pwd_col, label_col])
    df[pwd_col] = df[pwd_col].astype(str)
    df[label_col] = df[label_col].astype(int)
    
    df[label_col] = df[label_col] - df[label_col].min()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char", 
            ngram_range=(1, 5), 
            max_features=30_000,       
            sublinear_tf=True,
            dtype=np.float32 
        )),
        
        ("classifier", XGBClassifier(
            tree_method="hist",        
            device="cuda",             
            n_estimators=trees,      
            max_depth=7,               
            max_bin=64,                
            learning_rate=0.05,        
            subsample=0.8,             
            colsample_bytree=0.8,      
            grow_policy="lossguide",   
            random_state=42
        ))
    ])

    pipeline.fit(df[pwd_col], df[label_col])
    joblib.dump(pipeline, file)
    
    return pipeline

def predict_password(model, password: str):
    if not password:
        return -1, "Empty"

    pred_tier = int(model.predict([password])[0])
    
    if pred_tier == 0:
        label = "Weak"
    elif pred_tier == 1:
        label = "Average"
    else:
        label = "Strong"

    return pred_tier, label

if __name__ == "__main__":
    if os.path.exists(file):
        trained_model = joblib.load(file)
    else:
        trained_model = train_and_save_model()
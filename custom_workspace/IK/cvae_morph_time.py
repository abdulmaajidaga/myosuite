import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_data = os.path.dirname(BASE_DIR)

STROKE_DIR = os.path.join(root_data, "data/kinematic/Stroke")
HEALTHY_DIR = os.path.join(root_data, "data/kinematic/Healthy")
SCORES_FILE = os.path.join(BASE_DIR, "output/scores.csv")
MODEL_OUT = os.path.join(BASE_DIR, "output/duration_model.pkl")

def load_scores_map(filepath):
    try:
        df = pd.read_csv(filepath)
        id_col, score_col = df.columns[0], df.columns[1]
        df[id_col] = df[id_col].astype(str).str.replace('.mot', '', regex=False).str.strip()
        return dict(zip(df[id_col], df[score_col]))
    except: return {}

def main():
    print("--- Training Duration Predictor ---")
    score_map = load_scores_map(SCORES_FILE)
    
    X = [] # FMA Scores
    y = [] # Duration (Frames)
    
    # 1. Collect Stroke Data
    print("Scanning Stroke data...")
    for f in glob.glob(os.path.join(STROKE_DIR, "*.csv")):
        name = os.path.basename(f).replace('_processed.csv', '').replace('.csv', '')
        
        # Find Score
        score = score_map.get(name)
        if not score:
            for k in score_map:
                if k in name: score = score_map[k]; break
        
        if score:
            df = pd.read_csv(f)
            duration = len(df)
            X.append(int(score))
            y.append(duration)
            
    # 2. Collect Healthy Data (FMA = 66)
    print("Scanning Healthy data...")
    for f in glob.glob(os.path.join(HEALTHY_DIR, "*.csv")):
        df = pd.read_csv(f)
        duration = len(df)
        X.append(66) # Healthy is always max score
        y.append(duration)
        
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    
    print(f"Dataset Size: {len(X)} samples")
    
    # 3. Train Model (Polynomial Regression is best for non-linear biological trends)
    # Degree 2 captures the curve (Time drops quickly then plateaus)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    
    # 4. Save
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Model saved to {MODEL_OUT}")
    
    # 5. Visualize the Learned Curve
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color='blue', alpha=0.5, label='Real Data')
    
    # Plot prediction line
    test_x = np.linspace(10, 70, 100).reshape(-1, 1)
    pred_y = model.predict(test_x)
    
    plt.plot(test_x, pred_y, color='red', linewidth=2, label='Learned Trend')
    plt.xlabel('FMA Score')
    plt.ylabel('Duration (Frames)')
    plt.title('Learned Relationship: FMA vs Speed')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(BASE_DIR, "output/duration_curve.png"))
    print("Visualization saved to output/duration_curve.png")

if __name__ == "__main__":
    main()
import os
import pickle
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_PATH = DATA_DIR / "telemetry_dataset.csv"
MODEL_EXPORT_PATH = MODELS_DIR / "telemetry_rf_v1.pkl"

def train_telemetry_model():
    print("[ML Train RF] Loading dataset...")
    if not DATASET_PATH.exists():
        print(f"[ML Train RF] Error: Dataset not found at {DATASET_PATH}. Please run generate_telemetry.py first.")
        return False
        
    df = pd.read_csv(DATASET_PATH)
    
    # Split features and labels
    X = df[["time_spent_sec", "run_count", "backspace_count", "paste_char_count", "syntax_error_count"]]
    y = df["label"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"[ML Train RF] Training Random Forest classifier on {X_train.shape[0]} samples...")
    # Initialize and train classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Normal", "Shotgun Debugging", "Copy-Paste Dependency"])
    print("[ML Train RF] Evaluation Report:\n", report)
    
    # Check Quality Gate
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"[ML Train RF] Weighted F1-Score: {f1:.4f}")
    
    if f1 < 0.92:
        print("[ML Train RF] Warning: F1-Score is below quality gate of 0.92!")
    else:
        print("[ML Train RF] Quality gate PASSED (F1-Score > 0.92).")
        
    # Export model binary
    print(f"[ML Train RF] Exporting trained model binary to: {MODEL_EXPORT_PATH}")
    with open(MODEL_EXPORT_PATH, "wb") as f:
        pickle.dump(rf, f)
        
    print("[ML Train RF] Model training and binary export successfully complete!")
    return True

if __name__ == "__main__":
    train_telemetry_model()

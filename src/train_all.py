import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
from utils.logger import logger
import shutil

# Set up paths
SRC_DIR = Path(__file__).resolve().parent
ML_TRAINING_DIR = SRC_DIR.parent
DATA_DIR = ML_TRAINING_DIR / "data"
MODELS_DIR = ML_TRAINING_DIR / "models"

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Locate Python Engine models directory for copying models
ENGINE_MODELS_DIR = None
curr_dir = SRC_DIR
for _ in range(5):
    check_path = curr_dir / "services" / "engine-python" / "models"
    if check_path.exists():
        ENGINE_MODELS_DIR = check_path
        break
    # Check if we are already in the engine-python or similar peer folders
    peer_path = curr_dir.parent / "engine-python" / "models"
    if peer_path.exists():
        ENGINE_MODELS_DIR = peer_path
        break
    if curr_dir == curr_dir.parent:
        break
    curr_dir = curr_dir.parent

def load_data(file_name):
    path = DATA_DIR / file_name
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    logger.info(f"Loaded dataset: {file_name}")
    return pd.read_csv(path)

def train_and_evaluate_models(df, target_col, dataset_name):
    logger.info(f"--- Training and Comparing Models for {dataset_name} ---")
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Train-test split (stratified since it's a classification problem)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features for algorithms that are sensitive to scale (SVM, Logistic Regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models to compare
    models = {
        "RandomForest": (
            RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42), 
            X_train, X_test
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42), 
            X_train, X_test
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42), 
            X_train, X_test
        ),
        "DecisionTree": (
            DecisionTreeClassifier(max_depth=8, random_state=42), 
            X_train, X_test
        ),
        "LogisticRegression": (
            LogisticRegression(max_iter=1000, random_state=42), 
            X_train_scaled, X_test_scaled
        ),
        "SVM": (
            SVC(probability=True, random_state=42), 
            X_train_scaled, X_test_scaled
        )
    }
    
    results = {}
    best_f1 = -1.0
    best_model_name = ""
    best_model_obj = None
    best_scaler = None
    
    for name, (clf, x_tr, x_te) in models.items():
        try:
            logger.info(f"Training {name}...")
            clf.fit(x_tr, y_train)
            
            # Predict & evaluate
            y_pred = clf.predict(x_te)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            
            results[name] = {
                "accuracy": acc,
                "f1_score": f1,
                "report": classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info(f"{name} Results - Accuracy: {acc:.4f}, Weighted F1: {f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model_name = name
                best_model_obj = clf
                best_scaler = scaler if name in ["LogisticRegression", "SVM"] else None
                
        except Exception as e:
            logger.error(f"Error training {name} for {dataset_name}: {e}")
            
    logger.info(f"Best model for {dataset_name}: {best_model_name} with Weighted F1: {best_f1:.4f}")
    return best_model_obj, best_scaler, best_model_name, best_f1, results

def main():
    datasets = {
        "MCQ": ("mcq_nuanced_telemetry.csv", "target_class", "telemetry_mcq_model.pkl"),
        "Code": ("code_nuanced_telemetry.csv", "target_class", "telemetry_code_model.pkl"),
        "OCR": ("ocr_nuanced_telemetry.csv", "target_class", "telemetry_ocr_model.pkl")
    }
    
    overall_summary = {}
    
    for task_name, (file_name, target_col, model_file_name) in datasets.items():
        try:
            df = load_data(file_name)
            
            best_model, scaler, model_name, f1, results = train_and_evaluate_models(
                df, target_col, task_name
            )
            
            # Save the best model and optional scaler
            model_export_path = MODELS_DIR / model_file_name
            model_payload = {
                "model": best_model,
                "scaler": scaler,
                "features": list(df.drop(columns=[target_col]).columns),
                "classes": list(best_model.classes_) if hasattr(best_model, "classes_") else None
            }
            
            with open(model_export_path, "wb") as f:
                pickle.dump(model_payload, f)
            logger.info(f"Exported best {task_name} model to {model_export_path}")
            
            # Copy to python inference worker models directory if found
            if ENGINE_MODELS_DIR:
                os.makedirs(ENGINE_MODELS_DIR, exist_ok=True)
                dest_path = ENGINE_MODELS_DIR / model_file_name
                with open(dest_path, "wb") as f:
                    pickle.dump(model_payload, f)
                logger.info(f"Copied {task_name} model to Python Engine: {dest_path}")
            else:
                logger.warn("Could not find services/engine-python/models folder to copy model.")
                
            overall_summary[task_name] = {
                "best_model": model_name,
                "best_f1": f1,
                "all_results": results
            }
            
        except Exception as e:
            logger.error(f"Failed to process task {task_name}: {e}")
            
    # Print clean Markdown summary of performance
    print("\n" + "="*50)
    print("           MODEL COMPARISON SUMMARY")
    print("="*50)
    for task, info in overall_summary.items():
        print(f"\nTask: {task} Telemetry Classification")
        print(f"Chosen Best Model: {info['best_model']} (F1-score: {info['best_f1']:.4f})")
        print("-" * 40)
        print(f"{'Algorithm':<20} | {'Accuracy':<10} | {'F1-Score':<10}")
        print("-" * 40)
        for alg, metrics in info["all_results"].items():
            print(f"{alg:<20} | {metrics['accuracy']:<10.4f} | {metrics['f1_score']:<10.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()

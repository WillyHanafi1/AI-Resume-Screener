import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score
)
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """Train multiple scikit-learn models and compare performance."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
        'SVM': SVC(kernel='rbf', probability=True),
    }
    
    results = {}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_test': y_test,
        }
        
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    return results

def plot_confusion_matrix(y_test, y_pred, labels, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn."""
    os.makedirs('models', exist_ok=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'models/{title.replace(" ", "_").lower()}.png')
    plt.close()

def save_best_model(results: dict, filepath: str):
    """Save the best performing model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best_model = results[best_name]['model']
    joblib.dump(best_model, filepath)
    print(f"\nBest model: {best_name} (F1: {results[best_name]['f1_score']:.4f})")
    print(f"Saved to: {filepath}")
    return best_name, best_model

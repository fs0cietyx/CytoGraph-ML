import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def train_and_evaluate():
    print("Loading data for training...")
    X = pd.read_csv('data/data.csv', index_col=0)
    y = pd.read_csv('data/labels.csv', index_col=0)
    
    # Flatten y to a 1D array
    y = y.values.ravel()
    
    print("Splitting data into train and test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training Random Forest with 100 estimators on {X.shape[1]} features...")
    # Using n_jobs=-1 to use all available cores
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    print("Making predictions...")
    y_pred = rf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Extracting feature importance...")
    importances = rf.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({'Gene': feature_names, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Save the top 100 features
    os.makedirs('results', exist_ok=True)
    feature_importance_df.head(100).to_csv('results/top_100_features.csv', index=False)
    print("Top 100 features saved to results/top_100_features.csv")
    
    # Display the top 20
    print("\n--- Top 20 Most Predictive Genes ---")
    print(feature_importance_df.head(20))
    
    return feature_importance_df.head(20)

if __name__ == "__main__":
    train_and_evaluate()

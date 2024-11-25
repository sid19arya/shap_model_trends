import os
from src.data_loader import load_and_split_data
from src.train_model import train_model
from src.evaluate_model import evaluate_model
from src.shap_analysis import shap_analysis

DATA_PATH = "data/heart.csv"
OUTPUT_DIR = "plots/shap_feature_plots/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def main():
    print("Loading and splitting data...")
    data_by_hospital = load_and_split_data(DATA_PATH)
    models = {}
    scalers = {}
    
    print("Training models...")
    for hospital, data in data_by_hospital.items():
        X = data.drop(['id', 'num'], axis=1).values  # Drop ID, target, and dataset columns
        y = data['num'].values  # Target column
        model, scaler = train_model(X, y, input_size=X.shape[1])
        models[hospital] = model
        scalers[hospital] = scaler
    
    print("Evaluating models...")
    # Evaluate models on other hospitals
    for test_hospital, test_data in data_by_hospital.items():
        print(f"Testing models on {test_hospital}")
        for train_hospital, model in models.items():
            X_test = test_data.drop(['id', 'num'], axis=1).values
            y_test = test_data['num'].values
            accuracy, _ = evaluate_model(model, X_test, y_test, scalers[train_hospital])
            print(f"Model trained on {train_hospital} -> Test on {test_hospital}: Accuracy = {accuracy:.2f}")
    
    print("Performing SHAP analysis...")
    # SHAP analysis
    for hospital, data in data_by_hospital.items():
        X = data.drop(['id', 'num'], axis=1).values
        feature_names = data.drop(['id', 'num'], axis=1).columns
        shap_analysis(models[hospital], X, scalers[hospital], feature_names, OUTPUT_DIR, hospital)

if __name__ == "__main__":
    main()

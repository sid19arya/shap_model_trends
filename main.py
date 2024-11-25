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
    data_by_hospital = load_and_split_data(DATA_PATH)
    models = {}
    scalers = {}
    
    for hospital, data in data_by_hospital.items():
        X = data.drop(['target', 'hospital'], axis=1).values
        y = data['target'].values
        model, scaler = train_model(X, y, input_size=X.shape[1])
        models[hospital] = model
        scalers[hospital] = scaler
    
    # Evaluate models on other hospitals
    for test_hospital, test_data in data_by_hospital.items():
        print(f"Testing models on {test_hospital}")
        for train_hospital, model in models.items():
            X_test = test_data.drop(['target', 'hospital'], axis=1).values
            y_test = test_data['target'].values
            accuracy, _ = evaluate_model(model, X_test, y_test, scalers[train_hospital])
            print(f"Model trained on {train_hospital} -> Test on {test_hospital}: Accuracy = {accuracy:.2f}")
    
    # SHAP analysis
    for hospital, data in data_by_hospital.items():
        X = data.drop(['target', 'hospital'], axis=1).values
        feature_names = data.drop(['target', 'hospital'], axis=1).columns
        shap_analysis(models[hospital], X, scalers[hospital], feature_names, OUTPUT_DIR, hospital)

if __name__ == "__main__":
    main()

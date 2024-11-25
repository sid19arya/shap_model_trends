import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X, scaler, feature_names, output_dir, hospital_name):
    X_scaled = scaler.transform(X)
    explainer = shap.Explainer(model, X_scaled)
    shap_values = explainer(X_scaled)

    # Summary plot
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names, show=False)
    plt.savefig(f"{output_dir}/{hospital_name}_shap_summary.png")
    plt.close()
    
    # Key features plot
    for feature in feature_names[:5]:  # Top 5 features
        shap.dependence_plot(feature, shap_values, X_scaled, feature_names=feature_names, show=False)
        plt.savefig(f"{output_dir}/{hospital_name}_{feature}_dependence.png")
        plt.close()

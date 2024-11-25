import pandas as pd

def load_and_split_data(filepath):
    """Load the dataset and split it by hospitals."""
    df = pd.read_csv(filepath)
    hospitals = df['hospital'].unique()
    data_by_hospital = {hospital: df[df['hospital'] == hospital] for hospital in hospitals}
    return data_by_hospital

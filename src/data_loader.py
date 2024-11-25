import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def preprocess_data(df):
    """Preprocess data by encoding categorical variables and scaling features."""
    categorical_columns = ['sex', 'restecg']

    # Drop specified columns: slope, ca, thal
    df = df.drop(columns=['slope', 'ca', 'thal'])
    # Drop rows with NaN values
    df = df.dropna()

    df['fbs'] = df['fbs'].astype(int)  # Convert TRUE/FALSE to 1/0
    df['exang'] = df['exang'].astype(int)  # Convert TRUE/FALSE to 1/0
    
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
    encoded = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Concatenate encoded variables and drop original categorical columns
    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    return df, encoder

def load_and_split_data(filepath):
    """Load the dataset, preprocess it, and split by hospitals (datasets)."""
    df = pd.read_csv(filepath)
    df, encoder = preprocess_data(df)
    hospitals = df['dataset'].unique()
    data_by_hospital = {hospital: df[df['dataset'] == hospital].drop(columns=['dataset']) for hospital in hospitals}
    return data_by_hospital

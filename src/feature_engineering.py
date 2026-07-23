import os
import yaml
import pandas as pd
import pickle

def engineer_features(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config["data"]["processed_dir"]
    input_path = os.path.join(processed_dir, "cleaned_dataset.csv")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cleaned data not found at {input_path}")
        
    df = pd.read_csv(input_path)
    
    # Feature engineering
    df['RSRP_SINR_Ratio'] = df['RSRP'] / (df['SINR'] + 1e-5)
    df['Signal_Strength'] = pd.cut(df['RSRP'], bins=[-140, -100, -80, -60, 0], labels=[0, 1, 2, 3]).astype(int)
    
    features = ['RSRP', 'SINR', 'prev_beam', 'distance', 'RSRP_SINR_Ratio', 'Signal_Strength']
    target = 'best_beam'
    
    X = df[features]
    y = df[target]
    
    # Normalizer
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    columns_to_scale = ['RSRP', 'SINR', 'distance', 'RSRP_SINR_Ratio']
    X.loc[:, columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    
    # Save the scaler for inference
    scaler_path = os.path.join(processed_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    
    output_path_X = os.path.join(processed_dir, "X.csv")
    output_path_y = os.path.join(processed_dir, "y.csv")
    
    X.to_csv(output_path_X, index=False)
    y.to_csv(output_path_y, index=False)
    
    print(f"Features engineered and saved. Scaler saved at {scaler_path}")
    return X, y

if __name__ == "__main__":
    engineer_features()

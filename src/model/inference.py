import os
import yaml
import pickle
import pandas as pd

class BeamPredictor:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        # load chosen model (XGBoost)
        model_path = self.config["models"]["xgboost"]
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        # load scaler
        scaler_path = os.path.join(self.config["data"]["processed_dir"], "scaler.pkl")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
            
    def predict(self, rsrp, sinr, prev_beam, distance):
        """
        Inference logic for a single sample.
        """
        # Feature Engineering on the fly
        rsrp_sinr_ratio = rsrp / (sinr + 1e-5)
        
        # signal strength binning
        if rsrp <= -100: sig_strength = 0
        elif rsrp <= -80: sig_strength = 1
        elif rsrp <= -60: sig_strength = 2
        else: sig_strength = 3
            
        data = pd.DataFrame([{
            'RSRP': float(rsrp),
            'SINR': float(sinr),
            'prev_beam': int(prev_beam),
            'distance': float(distance),
            'RSRP_SINR_Ratio': float(rsrp_sinr_ratio),
            'Signal_Strength': int(sig_strength)
        }])
        
        # Scale
        cols_to_scale = ['RSRP', 'SINR', 'distance', 'RSRP_SINR_Ratio']
        data.loc[:, cols_to_scale] = self.scaler.transform(data[cols_to_scale])
        
        prediction = self.model.predict(data)
        return int(prediction[0])

if __name__ == "__main__":
    predictor = BeamPredictor()
    # Sample Test
    pred_beam = predictor.predict(rsrp=-90, sinr=15, prev_beam=3, distance=500)
    print(f"Predicted Optimal Beam: {pred_beam}")

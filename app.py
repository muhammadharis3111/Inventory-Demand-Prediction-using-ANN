from flask import Flask, request, jsonify, send_file
import torch
import numpy as np
import torch.nn as nn

class DemandANN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))
    def forward(self, x): 
        return self.net(x)

ckpt = torch.load('demand_ann_model.pth', map_location='cpu', weights_only=False)
model = DemandANN(ckpt['input_dim'])
model.load_state_dict(ckpt['model_state'])
model.eval()

s_mean = np.array(ckpt['scaler_mean'])
s_scale = np.array(ckpt['scaler_scale'])

app = Flask(__name__)

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    d = request.json
    f = np.array([[float(d['inventory_level']), float(d['units_sold']),
                   float(d['units_ordered']), float(d['price']),
                   float(d['discount']), int(d['promotion']),
                   float(d['competitor_pricing']), int(d['epidemic']),
                   int(d.get('category',0)), int(d.get('region',0)),
                   int(d.get('weather',0)), int(d.get('season',0))]])
    
    f = (f - s_mean) / s_scale
    
    with torch.no_grad():
        pred = model(torch.FloatTensor(f)).item()
        
    return jsonify({
        'predicted_demand': round(pred, 2),
        'model_r2': round(ckpt['r2_score'], 4),
        'model_rmse': round(ckpt['rmse'], 2)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
import ray
from ray import serve
from starlette.requests import Request
from pathlib import Path
import os
import time
import torch 
from torch.utils.data import DataLoader
from torch.ao.quantization import quantize_dynamic


BASE_DIR = Path(__file__).resolve().parents[1]
os.chdir(BASE_DIR)

from chestxray_module.dataset import data_load, transform
from chestxray_module.modeling.predict import load_model, predict, adjust, resolve_image_paths

ray.init(ignore_reinit_error=True)
serve.start()


@serve.deployment
class HealthService:
    def __init__(self):
        self.model = load_model("models/best_model.pt")
        self.model = quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = torch.compile(self.model)
        print(">>> Model compiled with torch.compile")

    async def __call__(self, request: Request):
        path = request.query_params.get("path")
        
        if request.url.path == "/":
            return {
                "status": "running",
                "model_loaded": True
                }
            
        if request.url.path == "/run_inference":
            if not path:
                return {"error": "path parameter is required"}
            
            
            print("preparing image(s)")
            path = resolve_image_paths(path)
            raw_data = data_load(data_dir=str(path), recursive=False, inspect=False)
            transformed_data = transform(raw_data, "test")
            adjusted_data = adjust(transformed_data)
            
            dataloader = DataLoader(
                adjusted_data,
                batch_size=1,
                shuffle=False,
                num_workers=0,
                #pin_memory=True,
            )
            
            print("running inference")
            preds, probs, paths = predict(self.model, dataloader)
            
            
            IDX_TO_CLASS = {
             0: "normal",
             1: "pneumonia",
             2: "tuberculosis",
            }
            
            results = []

            for p, prob, path in zip(preds, probs, paths):
                    results.append({
                    "path": str(path),
                    "predicted_class_index": int(p),
                    "predicted_class": IDX_TO_CLASS[int(p)],
                    "probabilities": {
                        IDX_TO_CLASS[i]: float(prob[i])
                        for i in range(len(prob))
                    }
                })

            return {
                "results": results
            }

            
        return {"error": "Unknown route"}
    

serve.run(HealthService.bind(), route_prefix="/")

while True:
    time.sleep(3600)


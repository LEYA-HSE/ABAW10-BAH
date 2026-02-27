import torch
import numpy as np
from decord import VideoReader, cpu
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

class VideoPredictor:
    def __init__(self, model_path='best_model.pth', config_dir='./'):
        """
        model_path: путь к .pth файлу с весами
        config_dir: папка, где лежит preprocessor_config.json
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.image_processor = VideoMAEImageProcessor.from_pretrained(config_dir)
      
        self.model = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.classes = {0: "Неуверенность", 1: "Уверенность"}

    def _get_video_frames(self, video_path, num_frames=16):
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
        frames = vr.get_batch(indices).asnumpy()
        return list(frames)

    def predict(self, video_path):

        frames = self._get_video_frames(video_path)
        inputs = self.image_processor(frames, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            conf, pred_idx = torch.max(probs, dim=-1)
            
        result = {
            "label": self.classes[pred_idx.item()],
            "confidence": conf.item(),
            "probs": probs.cpu().numpy().tolist()
        }
        return result


# Создаем объект предиктора 
predictor = VideoPredictor(model_path='/content/best_model.pth', config_dir='/content/')

# Делаем предсказание
video_file = "/content/0123.mp4"
prediction = predictor.predict(video_file)
# Если вывести prediction, то получаем итог: уверенность/неуверенность и вероятность классов


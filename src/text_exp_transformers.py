from transformers import AutoTokenizer, AutoModel
import torch

from src.text_core import save_experiment_predictions, calc_scores


def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()

model_name = "michellejieli/emotion_text_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# 3. Применение к колонке
df['embedding'] = df['text'].apply(get_embedding)

import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel


CKPT_PATH = "name.pt"
W2V_MODEL = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.init_weights()

    def forward(self, input_values):
        out = self.wav2vec2(input_values, output_hidden_states=True)
        return out.hidden_states


def _import_mamba_v1():
    try:
        from mamba_ssm.modules.mamba_simple import Mamba  # type: ignore
        return Mamba
    except Exception:
        from mamba_ssm import Mamba  # type: ignore
        return Mamba


class _MambaStack(nn.Module):
    def __init__(self, d_model, num_layers, d_state, d_conv, expand, dropout):
        super().__init__()
        Mamba = _import_mamba_v1()
        self.layers = nn.ModuleList(
            [Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(int(num_layers))]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(int(num_layers))])
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x, mask=None):
        for layer, ln in zip(self.layers, self.norms):
            h = layer(x)
            x = ln(x + self.drop(h))
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class MambaSequenceEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, dropout, pooling="mean",
                 mamba_d_state=16, mamba_d_conv=4, mamba_expand=2, **_unused):
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(d_model)) if int(input_dim) != int(d_model) else nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.pooling = str(pooling).lower()
        if self.pooling != "mean":
            raise ValueError(f"Unsupported pooling for this inference script: {self.pooling}")
        self.stack = _MambaStack(
            d_model=int(d_model),
            num_layers=int(num_layers),
            d_state=int(mamba_d_state),
            d_conv=int(mamba_d_conv),
            expand=int(mamba_expand),
            dropout=float(dropout),
        )
        self.out_dim = int(d_model)

    def forward(self, x, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
            if mask is None:
                mask = torch.ones(x.size(0), 1, dtype=torch.bool, device=x.device)

        x = self.input_proj(x)
        x = self.dropout(x)

        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)

        h = self.stack(x, mask=mask)

        if mask is None:
            return h.mean(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(h.dtype)
        return (h * mask.unsqueeze(-1)).sum(dim=1) / denom


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout):
        super().__init__()
        hidden = max(64, int(in_dim) // 2)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, int(num_classes)),
        )

    def forward(self, x):
        return self.net(x)


class BAHClassifier(nn.Module):
    def __init__(self, mcfg: dict, input_dim: int):
        super().__init__()
        self.encoder = MambaSequenceEncoder(
            input_dim=input_dim,
            d_model=int(mcfg["d_model"]),
            num_layers=int(mcfg["num_layers"]),
            dropout=float(mcfg["dropout"]),
            pooling=str(mcfg.get("pooling", "mean")),
            mamba_d_state=int(mcfg.get("mamba_d_state", 16)),
            mamba_d_conv=int(mcfg.get("mamba_d_conv", 4)),
            mamba_expand=int(mcfg.get("mamba_expand", 2)),
        )
        self.head = MLPHead(
            self.encoder.out_dim,
            num_classes=int(mcfg.get("num_classes", 2)),
            dropout=float(mcfg["dropout"]),
        )

    def forward(self, x, mask=None):
        z = self.encoder(x, mask=mask)
        return self.head(z)


processor = Wav2Vec2Processor.from_pretrained(W2V_MODEL)
w2v = EmotionModel.from_pretrained(W2V_MODEL).to(DEVICE).eval()

ckpt = torch.load(CKPT_PATH, map_location="cpu")
cfg = ckpt["cfg"]
mcfg = cfg["model"]
clf = BAHClassifier(mcfg, input_dim=1024).to(DEVICE).eval()
clf.load_state_dict(ckpt["model_state"], strict=True)


@torch.inference_mode()
def predict_audio(audio_path: str):
    signal, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(signal, sampling_rate=sr, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(DEVICE)

    hidden_states = w2v(input_values)
    layer10 = hidden_states[10]  # (1, T, 1024)

    x = layer10.to(DEVICE)
    mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=DEVICE)

    logits = clf(x, mask=mask)
    probs = torch.softmax(logits, dim=-1)

    pred = int(torch.argmax(probs, dim=-1).item())
    prob = float(probs[0, pred].item())
    return {"pred": pred, "prob": prob, "probs": probs[0].detach().cpu().numpy()}


out = predict_audio("name.wav")
print(out)

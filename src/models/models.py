# coding: utf-8
import torch.nn as nn


class VectorMLP(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        dropout=0.1,
        out_features=128,
        num_classes=2,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(out_features, num_classes)
        self._init_weights()

    def forward(self, x, mask=None, features: bool = False):
        if x.ndim == 3:
            x = x.squeeze(1)
        feats = self.feature_extractor(x)
        logits = self.classifier(feats)
        if features:
            return {"prob": logits, "features": feats}
        return logits

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

import torch
from torch import nn


class BiGRUPriorGuidedTail(nn.Module):
    def __init__(
        self,
        input_dim=29,
        hidden_dim=256,
        num_layers=2,
        dropout=0.15,
        bidirectional=True,
        output_dim=10,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.prior_head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        guided_dim = out_dim + 2
        self.correction_head = nn.Sequential(
            nn.LayerNorm(guided_dim),
            nn.Linear(guided_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.trust_head = nn.Sequential(
            nn.LayerNorm(guided_dim),
            nn.Linear(guided_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.yolo_error_bucket_head = nn.Sequential(
            nn.LayerNorm(guided_dim),
            nn.Linear(guided_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.yolo_quality_head = nn.Sequential(
            nn.LayerNorm(guided_dim),
            nn.Linear(guided_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        prior_tail = self.prior_head(out)
        prior_offset = prior_tail - x[:, :, 2:4]
        guided = torch.cat([out, prior_offset], dim=-1)
        delta_px = self.correction_head(guided)
        trust_logit = self.trust_head(guided)
        bucket_logits = self.yolo_error_bucket_head(guided)
        quality_logit = self.yolo_quality_head(guided)
        return torch.cat([prior_tail, delta_px, trust_logit, bucket_logits, quality_logit], dim=-1)


BiGRUTailDirect = BiGRUPriorGuidedTail
BiGRUTailGate = BiGRUPriorGuidedTail


def build_model(
    model_type,
    input_dim=29,
    hidden_dim=256,
    num_layers=2,
    dropout=0.15,
    bidirectional=True,
    output_dim=10,
):
    model_type = model_type.lower()
    if model_type in {"bigru", "gru"}:
        if output_dim != 10:
            raise ValueError(f"v8 prior-guided model requires output_dim=10, got {output_dim}")
        return BiGRUPriorGuidedTail(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            output_dim=output_dim,
        )
    raise ValueError(f"Unknown model type: {model_type}")

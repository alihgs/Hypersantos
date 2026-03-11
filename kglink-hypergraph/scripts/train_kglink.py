from src.model.kglink_model import KGLinkModel
from transformers import AutoTokenizer
import torch

model = KGLinkModel(num_labels=50)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
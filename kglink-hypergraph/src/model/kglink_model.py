import torch.nn as nn
from transformers import AutoModel

class KGLinkModel(nn.Module):

    def __init__(self, plm="bert-base-uncased", num_labels=100):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(plm)

        hidden = self.encoder.config.hidden_size

        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):

        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask)

        cls = out.last_hidden_state[:, 0, :]

        logits = self.classifier(cls)

        return logits
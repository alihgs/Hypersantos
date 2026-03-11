from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.preprocessing.table_serialization import serialize_table

class KGLinkDataset(Dataset):

    def __init__(self, tables, labels, tokenizer="bert-base-uncased"):

        self.tables = tables
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def __len__(self):
        return len(self.tables)

    def __getitem__(self, idx):

        table = self.tables[idx]
        target = list(table.columns)[0]

        text = serialize_table(table, target)

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": self.labels[idx]
        }
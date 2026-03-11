import torch

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for batch in loader:

            logits = model(batch["input_ids"], batch["attention_mask"])

            preds = logits.argmax(dim=-1)

            correct += (preds == batch["label"]).sum().item()

            total += batch["label"].size(0)

    return correct / total
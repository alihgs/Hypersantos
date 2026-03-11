import torch

def train(model, loader, optimizer, loss_fn):

    model.train()

    total = 0

    for batch in loader:

        logits = model(batch["input_ids"], batch["attention_mask"])

        loss = loss_fn(logits, batch["label"])

        loss.backward()

        optimizer.step()

        optimizer.zero_grad()

        total += loss.item()

    return total
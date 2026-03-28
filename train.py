import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TextDataset
from model import GPT


dataset = TextDataset()

loader = DataLoader(
    dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

model = GPT().to(config.device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate
)


step = 0

for epoch in range(1000):

    for x, y in loader:

        x = x.to(config.device)
        y = y.to(config.device)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("step", step, "loss", loss.item())

        step += 1

        if step >= config.max_iters:
            break

    if step >= config.max_iters:
        break


torch.save(model.state_dict(), "model.pt")
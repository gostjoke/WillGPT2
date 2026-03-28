import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import TextDataset
from model import GPT


def main():

    print("torch version:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = TextDataset()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    model = GPT().to(config.device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate
    )

    # mixed precision (提升速度)
    scaler = torch.cuda.amp.GradScaler()

    step = 0

    pbar = tqdm(total=config.max_iters)

    for epoch in range(1000):

        for x, y in loader:

            x = x.to(config.device, non_blocking=True)
            y = y.to(config.device, non_blocking=True)

            with torch.cuda.amp.autocast():

                logits, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            if step % 100 == 0:
                print("step", step, "loss", loss.item())

            # checkpoint
            if step % 1000 == 0 and step > 0:
                torch.save(model.state_dict(), f"checkpoint_{step}.pt")

            step += 1
            pbar.update(1)

            if step >= config.max_iters:
                break

        if step >= config.max_iters:
            break

    pbar.close()

    torch.save(model.state_dict(), f"WillGPT_v{config.model_save_version}.pt.pt")

    print("training finished")


if __name__ == "__main__":
    main()
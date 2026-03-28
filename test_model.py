from model import GPT
import torch

model = GPT()

x = torch.randint(0,5000,(4,128))

logits,loss = model(x,x)

print(logits.shape)
print(loss)
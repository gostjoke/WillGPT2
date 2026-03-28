"""
Text Generation

The generate.py script is used to perform text generation with the trained GPT model.
After the model has been trained and saved as model.pt, this script loads the model and generates new text based on a user-defined prompt.

First, the script loads the trained model weights and moves the model to the configured device (CPU or CUDA GPU). The model is then switched to evaluation mode to disable training-specific behaviors such as dropout.

Next, the input prompt is tokenized using the GPT-2 tokenizer provided by the tiktoken library. The encoded tokens are converted into a PyTorch tensor and passed into the model.

The generation process works in an autoregressive manner. At each step:

The model predicts the probability distribution of the next token.
The probabilities are obtained using the softmax function.
A new token is sampled from this distribution.
The sampled token is appended to the existing sequence.
The updated sequence is fed back into the model for the next prediction.

This process is repeated for a fixed number of steps (e.g., 50 tokens), gradually extending the generated text. Finally, the resulting token sequence is decoded back into human-readable text and printed to the console.

This approach demonstrates the core idea behind modern large language models: generating text one token at a time based on previous context.
"""

import torch
import tiktoken

from model import GPT
import config

device = config.device

model = GPT().to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))

model.eval()

enc = tiktoken.get_encoding("gpt2")

prompt = input("Enter a prompt: ")

tokens = enc.encode(prompt)

x = torch.tensor(tokens).unsqueeze(0).to(device)

with torch.no_grad():

    for _ in range(50):

        logits, _ = model(x[:, -config.block_size:])

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

print(enc.decode(x[0].tolist()))
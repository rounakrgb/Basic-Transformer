import torch
import torch.nn as nn
import torch.optim as optim
import random
import string

# ----------------------------
# 1. Vocabulary
# ----------------------------
chars = list(string.ascii_lowercase)
vocab = {ch: i for i, ch in enumerate(chars)}
inv_vocab = {i: ch for ch, i in vocab.items()}
vocab_size = len(vocab)

def encode(word):
    return torch.tensor([vocab[c] for c in word], dtype=torch.long)

def decode(tensor):
    return "".join([inv_vocab[i.item()] for i in tensor])

# ----------------------------
# 2. Model
# ----------------------------
class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, max_len):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.positional_embedding = nn.Embedding(max_len, hidden_dim)

        self.self_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        positions = torch.arange(T, device=x.device)
        positions = positions.unsqueeze(0).expand(B, T)

        x = self.token_embedding(x) + self.positional_embedding(positions)

        attn_out, _ = self.self_attention(x, x, x)

        x = self.feed_forward(attn_out)

        logits = self.output_layer(x)

        return logits


# ----------------------------
# 3. Create Model
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    vocab_size=vocab_size,
    hidden_dim=64,
    num_heads=4,
    max_len=20
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ----------------------------
# 4. Generate Random Words
# ----------------------------
def random_word(min_len=3, max_len=8):
    length = random.randint(min_len, max_len)
    return "".join(random.choices(chars, k=length))

# ----------------------------
# 5. Training Loop
# ----------------------------
batch_size = 32
steps = 2000

for step in range(steps):

    words = [random_word() for _ in range(batch_size)]

    encoded = [encode(w) for w in words]
    targets = [encode(w[::-1]) for w in words]

    x = nn.utils.rnn.pad_sequence(encoded, batch_first=True).to(device)
    y = nn.utils.rnn.pad_sequence(targets, batch_first=True).to(device)

    logits = model(x)

    loss = criterion(
        logits.view(-1, vocab_size),
        y.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 200 == 0:
        print(f"Step {step} | Loss: {loss.item():.4f}")

test_word = "hii"

x_test = encode(test_word).unsqueeze(0).to(device)

with torch.no_grad():
    logits = model(x_test)
    pred = logits.argmax(dim=-1)

pred_word = decode(pred[0][:len(test_word)].cpu())


print("\nTest Word:", test_word)
print("Prediction:", pred_word)
print("Expected :", test_word[::-1])

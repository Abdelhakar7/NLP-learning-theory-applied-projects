import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# tokenize the corpus keeping only words
#returns a list of tokens 

def tokenize(text) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())
 
 # Read all text files from the corpus directory and tokenize them

def read_corpus(corpus_dir) -> list[str]:
    tokens = []
    for fname in os.listdir(corpus_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(corpus_dir, fname), encoding='utf-8') as f:
                tokens.extend(tokenize(f.read()))
    return tokens
# Heaps' Law: V = k * N^b
# N: total number of tokens, V: vocabulary size
# returns two numpy arrays: N (token counts) and V (vocabulary sizes)

def heaps_law(tokens, step=100) -> tuple[np.ndarray, np.ndarray]:
    vocab = set()
    N = []
    V = []
    for i in range(step, len(tokens)+1, step):
        vocab.update(tokens[i-step:i])
        N.append(i)
        V.append(len(vocab))
    return np.array(N), np.array(V)

def heaps_func(N, k, b):
    return k * (N ** b)

corpus_dir = './corpus'
tokens = read_corpus(corpus_dir)

# Heaps' Law data
N, V = heaps_law(tokens)

# Fit Heaps' Law: V = k * N^b
params, _ = curve_fit(heaps_func, N, V, p0=[1, 0.5])
k, b = params
t = len(N)
l= len (V)
#print(f"Token counts (N): {t}")
#print(f"Vocabulary sizes (V): {l}")
print(f"Heaps' Law parameters: k = {k:.4f}, b = {b:.4f}")
print(f"Total number of tokens (instances): {len(tokens)}")
print(f"Total number of distinct words (vocabulary): {len(set(tokens))}")

# Plot Heaps' Law
plt.figure(figsize=(6, 4))
plt.plot(N, V, 'o', label="Empirical data")
plt.plot(N, heaps_func(N, k, b), 'r-', label=f"Fit: V = {k:.2f} * N^{b:.2f}")
plt.title("Heaps' Law")
plt.xlabel("Total Tokens (N)")
plt.ylabel("Vocabulary Size (V)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("heaps_law.png")

# --- Zipf's Law analysis and plot ---
freq_counter = Counter(tokens)
frequencies = np.array(sorted(freq_counter.values(), reverse=True))
ranks = np.arange(1, len(frequencies) + 1)

plt.figure(figsize=(6, 4))
plt.loglog(ranks, frequencies, marker='.', linestyle='none', label="Empirical data")
plt.title("Zipf's Law")
plt.xlabel("Word Rank")
plt.ylabel("Word Frequency")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("zipf_law.png")
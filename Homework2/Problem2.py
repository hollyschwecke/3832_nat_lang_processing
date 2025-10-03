import nltk
import sentencepiece as spm 
import os

# download necessary nltk tokenizer
nltk.download("punkt")

file_path = os.path.join(os.path.dirname(__file__), "DorianGray.txt")
with open(file_path, 'r', encoding="utf-8") as file:
    text = file.read()
    
# part a: tokenize using nltk.word_tokenize
tokens_nltk = nltk.word_tokenize(text)
print("a) Number of unique tokens:", len(tokens_nltk))

# b) Unique tokens using set
unique_tokens_nltk = set(tokens_nltk)
print("b) Number of unique tokens (NLTK):", len(unique_tokens_nltk))

# Part (c) & (d): Train and use SentencePiece for BPE
# Only train once; comment this block if model already exists
spm.SentencePieceTrainer.train(
    input=file_path,
    model_prefix="bpe",
    vocab_size=2000,
    model_type="bpe",
    user_defined_symbols=["<eos>"]
)

# Load the trained SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load("bpe.model")

# Encode the original text using BPE
bpe_tokens = sp.encode(text, out_type=str)

# c) Total BPE tokens (non-unique)
print("c) Total number of BPE tokens:", len(bpe_tokens))

# d) Unique BPE tokens used
unique_bpe_tokens = set(bpe_tokens)
print("d) Number of unique BPE tokens used:", len(unique_bpe_tokens))

import json
import os
import tensorflow as tf
from datasets import load_dataset

START, END = "startseq", "endseq"  # plain-ASCII markers (survive TextVectorization standardization)


def prepare_iitb_dataset(subset_size=200000, output_dir="iitb_corpus"):
    """Downloads a subset of the IIT Bombay dataset from Hugging Face, 
    saves text files locally, and returns the file paths."""
    os.makedirs(output_dir, exist_ok=True)
    en_path = os.path.join(output_dir, "train.en")
    hi_path = os.path.join(output_dir, "train.hi")
    
    if not os.path.exists(en_path) or not os.path.exists(hi_path):
        print(f"Downloading {subset_size} sentence pairs from Hugging Face...")
        dataset = load_dataset("cfilt/iitb-english-hindi", split=f"train[:{subset_size}]")
        
        print("Writing dataset subset to local text files...")
        with open(en_path, "w", encoding="utf-8") as f_en, \
             open(hi_path, "w", encoding="utf-8") as f_hi:
            for item in dataset:
                pair = item["translation"]
                f_en.write(pair["en"].strip() + "\n")
                f_hi.write(pair["hi"].strip() + "\n")
        print("Dataset successfully prepared!")
    else:
        print("Local dataset files already found. Skipping download.")

    return en_path, hi_path


def load_texts(en_path, hi_path):
    """Loads English and Hindi lines from text files into lists."""
    print("Loading texts into memory...")
    with open(en_path, "r", encoding="utf-8") as f:
        eng_texts = [line.strip() for line in f if line.strip()]
    with open(hi_path, "r", encoding="utf-8") as f:
        hin_texts = [line.strip() for line in f if line.strip()]
    return eng_texts, hin_texts


def create_vectorizers(train_eng_texts, train_hin_texts, max_vocab_size=32000,
                       sequence_length=50, batch_size=1024):
    eng_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size, output_mode='int', output_sequence_length=sequence_length)
    # Hindi gets +1 so that input (:-1) and target (1:) are both `sequence_length` long
    hin_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size, output_mode='int', output_sequence_length=sequence_length + 1)

    print("Adapting English vectorizer...")
    eng_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(list(train_eng_texts)).batch(batch_size))

    print("Adapting Hindi vectorizer...")
    hin_marked = [f"{START} {t} {END}" for t in train_hin_texts]
    hin_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(hin_marked).batch(batch_size))
    return eng_vectorizer, hin_vectorizer


def save_vocab(vectorizer, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vectorizer.get_vocabulary(), f, ensure_ascii=False)


def load_vectorizer(path, max_vocab_size, sequence_length):
    with open(path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    vec = tf.keras.layers.TextVectorization(
        max_tokens=max_vocab_size, output_mode='int',
        output_sequence_length=sequence_length, vocabulary=vocab[2:])  # drop '' and [UNK]; layer re-adds them
    return vec


def make_dataset(eng_texts, hin_texts, eng_vectorizer, hin_vectorizer, batch_size=64, seq_len=50):
    print("Building generator-based tf.data pipeline...")

    def generator():
        chunk_size = 10000
        for i in range(0, len(eng_texts), chunk_size):
            en_chunk = list(eng_texts[i:i + chunk_size])
            hi_chunk = [f"{START} {t} {END}" for t in hin_texts[i:i + chunk_size]]
            eng_tokens = eng_vectorizer(en_chunk).numpy()
            hin_tokens = hin_vectorizer(hi_chunk).numpy()
            for en_t, hi_t in zip(eng_tokens, hin_tokens):
                yield (en_t, hi_t[:-1]), hi_t[1:]

    output_signature = (
        (tf.TensorSpec(shape=(seq_len,), dtype=tf.int64),
         tf.TensorSpec(shape=(seq_len,), dtype=tf.int64)),
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int64),
    )
    ds = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

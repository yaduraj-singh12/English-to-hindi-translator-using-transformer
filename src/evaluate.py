import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from src.transformer import TransformerTranslationModel
from src.dataset import load_vectorizer, START, END

# MUST match train.py
MAX_VOCAB_SIZE, SEQUENCE_LENGTH = 32000, 50
D_MODEL, NUM_HEADS, NUM_LAYERS, DFF = 512, 8, 6, 2048


def translate_sentence(sentence, model, eng_vec, hin_vocab, max_length=50):
    """Greedy autoregressive decoding starting from the [start] token."""
    word_to_id = {w: i for i, w in enumerate(hin_vocab)}
    start_id, end_id = word_to_id[START], word_to_id[END]

    src = eng_vec([sentence])
    tgt = tf.constant([[start_id]], dtype=tf.int64)

    for _ in range(max_length):
        logits = model((src, tgt), training=False)
        next_id = int(tf.argmax(logits[0, -1, :]))
        if next_id == end_id or next_id == 0:
            break
        tgt = tf.concat([tgt, tf.constant([[next_id]], dtype=tf.int64)], axis=1)

    ids = tgt.numpy()[0][1:]
    return " ".join(hin_vocab[i] for i in ids if hin_vocab[i] not in ("", "[UNK]"))


def main():
    # Use the SAME vocabularies saved during training (not a different dataset slice)
    eng_vec = load_vectorizer("eng_vocab.json", MAX_VOCAB_SIZE, SEQUENCE_LENGTH)
    hin_vocab = [""] + ["[UNK]"] + __import__("json").load(open("hin_vocab.json", encoding="utf-8"))[2:]

    model = TransformerTranslationModel(NUM_LAYERS, D_MODEL, NUM_HEADS, DFF,
                                        MAX_VOCAB_SIZE, MAX_VOCAB_SIZE)
    model((tf.zeros((1, SEQUENCE_LENGTH), tf.int64), tf.zeros((1, SEQUENCE_LENGTH), tf.int64)))  # build
    model.load_weights("transformer_en_hi.weights.h5")

    for s in ["What is your name?", "How are you?", "Where are you going?"]:
        print(f"English: {s}\nHindi:   {translate_sentence(s, model, eng_vec, hin_vocab)}\n")


if __name__ == "__main__":
    main()
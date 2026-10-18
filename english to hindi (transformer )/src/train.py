import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from src.transformer import TransformerTranslationModel
from src.dataset import create_vectorizers, make_dataset, save_vocab, prepare_iitb_dataset


def load_real_data():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(project_root, "iitb_corpus")
    
    # Automatically download and slice the 200k subset if it doesn't exist yet
    prepare_iitb_dataset(subset_size=200000, output_dir=data_dir)
    
    en_file = os.path.join(data_dir, "train.en")
    hi_file = os.path.join(data_dir, "train.hi")

    eng_texts, hin_texts = [], []
    with open(en_file, 'r', encoding='utf-8') as f_en, open(hi_file, 'r', encoding='utf-8') as f_hi:
        for le, lh in zip(f_en, f_hi):
            le, lh = le.strip(), lh.strip()
            if le and lh:
                eng_texts.append(le)
                hin_texts.append(lh)
    print(f"Loaded {len(eng_texts):,} sentence pairs")
    return eng_texts, hin_texts


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Standard Transformer warmup; a flat 3e-4 usually diverges with post-LN."""
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = float(d_model)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(
            tf.math.rsqrt(step + 1), step * (self.warmup_steps ** -1.5))


def loss_function(real, preds):
    mask = tf.cast(tf.math.not_equal(real, 0), tf.float32)
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, preds, from_logits=True)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def train_model():
    eng_corpus, hin_corpus = load_real_data()

    MAX_VOCAB_SIZE, SEQUENCE_LENGTH = 32000, 50
    D_MODEL, NUM_HEADS, NUM_LAYERS, DFF = 512, 8, 6, 2048
    BATCH_SIZE, EPOCHS = 64, 10

    eng_vec, hin_vec = create_vectorizers(eng_corpus, hin_corpus, MAX_VOCAB_SIZE, SEQUENCE_LENGTH)
    save_vocab(eng_vec, "eng_vocab.json")   # needed by evaluate.py
    save_vocab(hin_vec, "hin_vocab.json")

    train_dataset = make_dataset(eng_corpus, hin_corpus, eng_vec, hin_vec,
                                 batch_size=BATCH_SIZE, seq_len=SEQUENCE_LENGTH)

    transformer = TransformerTranslationModel(
        num_layers=NUM_LAYERS, d_model=D_MODEL, num_heads=NUM_HEADS, dff=DFF,
        src_vocab_size=MAX_VOCAB_SIZE, tgt_vocab_size=MAX_VOCAB_SIZE, dropout_rate=0.1)

    optimizer = tf.keras.optimizers.Adam(WarmupSchedule(D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    @tf.function
    def train_step(src, tgt_input, tgt_expected):
        with tf.GradientTape() as tape:
            predictions = transformer((src, tgt_input), training=True)
            loss = loss_function(tgt_expected, predictions)
        grads = tape.gradient(loss, transformer.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, transformer.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        total_loss, batches = 0.0, 0
        for (src, tgt_input), tgt_expected in train_dataset:
            total_loss += float(train_step(src, tgt_input, tgt_expected))
            batches += 1
            if batches % 200 == 0:
                print(f"  epoch {epoch+1} batch {batches} loss {total_loss/batches:.4f}")
        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / batches:.4f}")
        transformer.save_weights("transformer_en_hi.weights.h5")   # save every epoch

    print("Done.")


if __name__ == "__main__":
    train_model()

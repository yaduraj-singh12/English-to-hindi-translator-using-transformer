import tensorflow as tf
from tensorflow.keras import layers
from src.layers import PositionalEncoding, EncoderLayer, DecoderLayer


class TransformerTranslationModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 src_vocab_size, tgt_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model

        self.src_embedding = layers.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = layers.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_len=200, d_model=d_model)
        self.emb_dropout = layers.Dropout(dropout_rate)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]

        self.final_layer = layers.Dense(tgt_vocab_size)

    def call(self, inputs, training=False):
        src, tgt = inputs
        src_len = tf.shape(src)[1]
        tgt_len = tf.shape(tgt)[1]

        # Masks: 1 = may attend, 0 = blocked
        src_pad = tf.cast(tf.not_equal(src, 0), tf.float32)[:, tf.newaxis, :]      # [B,1,S]
        tgt_pad = tf.cast(tf.not_equal(tgt, 0), tf.float32)[:, tf.newaxis, :]      # [B,1,T]
        causal = tf.linalg.band_part(tf.ones((tgt_len, tgt_len)), -1, 0)[tf.newaxis]  # [1,T,T]

        enc_mask = src_pad * tf.ones((1, src_len, 1))          # [B,S,S]
        look_ahead_mask = causal * tgt_pad                     # [B,T,T]
        cross_mask = src_pad * tf.ones((1, tgt_len, 1))        # [B,T,S]

        scale = tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.pos_encoding(self.src_embedding(src) * scale)
        x = self.emb_dropout(x, training=training)
        for enc in self.enc_layers:
            x = enc(x, training=training, mask=enc_mask)
        enc_output = x

        y = self.pos_encoding(self.tgt_embedding(tgt) * scale)
        y = self.emb_dropout(y, training=training)
        for dec in self.dec_layers:
            y = dec(y, enc_output, training=training,
                    look_ahead_mask=look_ahead_mask, padding_mask=cross_mask)

        return self.final_layer(y)
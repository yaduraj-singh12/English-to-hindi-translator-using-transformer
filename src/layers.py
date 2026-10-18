import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_len=200, d_model=512, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = self.positional_encoding(max_len, d_model)

    def get_angles(self, position, i, d_model):
        rates = 1 / tf.math.pow(10000.0, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model,
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, x):
        return x + self.pos_encoding[:, :tf.shape(x)[1], :]


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
    ])


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        # key_dim is PER HEAD -> d_model // num_heads (was d_model: 8x too big)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False, mask=None):
        attn = self.mha(x, x, x, attention_mask=mask)
        out1 = self.layernorm1(x + self.dropout1(attn, training=training))
        ffn = self.ffn(out1)
        return self.layernorm2(out1 + self.dropout2(ffn, training=training))


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training=False, look_ahead_mask=None, padding_mask=None):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        out1 = self.layernorm1(x + self.dropout1(attn1, training=training))

        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        out2 = self.layernorm2(out1 + self.dropout2(attn2, training=training))

        ffn = self.ffn(out2)
        return self.layernorm3(out2 + self.dropout3(ffn, training=training))
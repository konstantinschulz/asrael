from typing import List, Any
import tensorflow as tf
import numpy as np
from tensorflow import Tensor, keras
from tensorflow.python.data import Dataset
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import Mean, SparseCategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow_datasets.core.features.text import SubwordTextEncoder


class Cluster:
    def __init__(self, tensors: List[Tensor], average_tensor: Tensor = None):
        self.tensors: List[Tensor] = tensors
        self.average_tensor: Tensor = average_tensor

    @classmethod
    def from_json(cls, json_dict: dict):
        return Cluster(json_dict["tensors"], json_dict.get("average_tensor", None))

    def to_json(self) -> dict:
        ret_val: dict = dict()
        numpy_tensors: List[np.ndarray] = [x.numpy() for x in self.tensors]
        ret_val["tensors"] = [x.tolist() for x in numpy_tensors]
        if self.average_tensor:
            ret_val["average_tensor"] = self.average_tensor.numpy().tolist()
        return ret_val


class Context:
    def __init__(self, content: str, token_range: str):
        self.content: str = content
        range_parts: List[int] = [int(x) for x in token_range.split(":")]
        self.token_range_start: int = range_parts[0]
        self.token_range_end: int = range_parts[1]


class Example:
    def __init__(self, line: str):
        parts: List[str] = line.split("\t")
        self.context1: Context = Context(parts[0], parts[2])
        self.context2: Context = Context(parts[1], parts[3])
        if len(parts) > 4:
            self.label: int = int(parts[4])


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def get_config(self):
        pass

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = keras.backend.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * keras.backend.minimum(arg1, arg2)


class Decoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.dec_layers: List[DecoderLayer] = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, inputs: List[Any], **kwargs):
        x, enc_output, training, look_ahead_mask, padding_mask = inputs
        seq_len = keras.backend.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(keras.backend.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            dec_layer: DecoderLayer = self.dec_layers[i]
            x, block1, block2 = dec_layer([x, enc_output, training, look_ahead_mask, padding_mask])
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class DecoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)
        self.dropout3 = keras.layers.Dropout(rate)

    def call(self, inputs: List[Any], **kwargs):
        x, enc_output, training, look_ahead_mask, padding_mask = inputs
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1([x, x, x, look_ahead_mask])  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(
            [enc_output, enc_output, out1, padding_mask])  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        return out3, attn_weights_block1, attn_weights_block2


class Encoder(keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        self.enc_layers: List[EncoderLayer] = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = keras.layers.Dropout(rate)

    def call(self, inputs: List[Any], **kwargs):
        x, training, mask = inputs
        seq_len = keras.backend.shape(x)[1]
        # adding embedding and position encoding.
        x: Tensor = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(keras.backend.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        x = keras.backend.expand_dims(x, axis=0)
        for i in range(self.num_layers):
            enc_layer: EncoderLayer = self.enc_layers[i]
            # only give the output of the last layer to the next layer, not all the previous outputs
            layer_output: Tensor = enc_layer([x[-1], training, mask])
            layer_output_exp: Tensor = keras.backend.expand_dims(layer_output, axis=0)
            x = keras.backend.concatenate([x, layer_output_exp], axis=0)
            # at the last layer, overwrite all the previous output and only return the current output
            # x = enc_layer([x, training, mask])
        return x  # (batch_size, input_seq_len, d_model)


class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs: List[Any], **kwargs):
        x, training, mask = inputs
        attn_output, _ = self.mha([x, x, x, mask])  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2


class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = keras.layers.Dense(d_model)
        self.wk = keras.layers.Dense(d_model)
        self.wv = keras.layers.Dense(d_model)
        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = keras.backend.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs: List[Any], **kwargs):
        v, k, q, mask = inputs
        batch_size = keras.backend.shape(q)[0]
        q = self.wq(q)  # (layer_id, batch_size, seq_len, d_model)
        k = self.wk(k)  # (layer_id, batch_size, seq_len, d_model)
        v = self.wv(v)  # (layer_id, batch_size, seq_len, d_model)
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = keras.backend.reshape(scaled_attention,
                                                 (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


class Transformer(keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder: Encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder: Decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        self.final_layer: keras.layers.Dense = keras.layers.Dense(target_vocab_size)
        # only for custom computation of embeddings
        self.avg: keras.layers.Average = keras.layers.Average()
        self.loss_object: SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none')
        learning_rate: CustomSchedule = CustomSchedule(d_model)
        self.optimizer: OptimizerV2 = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        self.train_accuracy: SparseCategoricalAccuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_loss: Mean = keras.metrics.Mean(name='train_loss')

    def call(self, inputs: List[Any], **kwargs):
        inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask = inputs
        enc_output: Tensor = self.encoder(
            [inp, training, enc_padding_mask])  # (layer_id, batch_size, inp_seq_len, d_model)
        enc_output_last_layer: Tensor = enc_output[-1]
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            [tar, enc_output_last_layer, training, look_ahead_mask, dec_padding_mask])
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        return final_output, attention_weights

    def check_token_position(self, idx: int, sentence: str, token: str) -> bool:
        next_char_idx: int = idx + len(token)
        if not idx:
            if next_char_idx == len(sentence):
                return True
            return next_char_idx < len(sentence) and not sentence[next_char_idx].isalpha()
        else:
            if sentence[idx - 1].isalpha():
                return False
            elif next_char_idx == len(sentence):
                return True
            else:
                return not sentence[next_char_idx].isalpha()

    def compute_embeddings(self, tok_idx: int, token: str, enc_output_no_batch: Tensor, sentence: str,
                           input_tensor: Tensor, tokenizer: SubwordTextEncoder) -> List[Tensor]:
        found_tensors: List[Tensor] = []
        target_representation: List[int]
        next_idx: int = tok_idx + len(token)
        # add whitespace after the token if present in the input, otherwise use just the raw token
        if next_idx < len(sentence) and sentence[next_idx] == " " and not token.isupper():
            target_representation = tokenizer.encode(token + " ")
        else:
            target_representation = tokenizer.encode(token)
        cursor: int = 0
        found_indices: List[int] = []
        for i in range(len(input_tensor)):
            if input_tensor[i] == target_representation[cursor]:
                cursor += 1
                if cursor == len(target_representation):
                    found_indices.append(i - len(target_representation) + 1)
                    cursor = 0
            else:
                cursor = 0
        for found_idx in found_indices:
            local_indices: List[int] = [found_idx + i for i in range(len(target_representation))]
            layer_tensors: List[Tensor] = []
            # iterate over the outputs of the last 4 layers
            for layer_id in range(1, len(enc_output_no_batch)):
                local_tensors: List[Tensor] = [enc_output_no_batch[layer_id][i] for i in local_indices]
                found_tensor: Tensor = self.avg(local_tensors) if len(local_tensors) > 1 else local_tensors[0]
                layer_tensors.append(found_tensor)
            found_tensors.append(self.avg(layer_tensors))
        return found_tensors

    def get_config(self):
        pass

    def get_embeddings_for_token(self, sentence: str, tokenizer: SubwordTextEncoder, token: str) -> List[Tensor]:
        def generator():
            for x in [sentence]:
                yield x

        dataset: Dataset = tf.data.Dataset.from_generator(generator, tf.string, tf.TensorShape([]))
        sentence_tensor: Tensor = next(enumerate(dataset))[1]
        vs: int = tokenizer.vocab_size
        input_tensor: Tensor = [vs] + tokenizer.encode(sentence_tensor.numpy()) + [vs + 1]
        encoder_input: Tensor = keras.backend.expand_dims(input_tensor, 0)
        enc_padding_mask = create_padding_mask(encoder_input)
        enc_output: Tensor = self.encoder(
            [encoder_input, True, enc_padding_mask])  # (layer_id, batch_size, inp_seq_len, d_model)
        # batch size is 1, so might as well remove that dimension completely
        enc_output_no_batch: Tensor = keras.backend.squeeze(enc_output, 1)  # (layer_id, inp_seq_len, d_model)
        tok_idx: int = sentence.find(token)
        next_char_idx: int = tok_idx + len(token)
        # if the found string is part of a larger token, retry until we get a separate instance
        while tok_idx > -1 and not self.check_token_position(tok_idx, sentence, token):
            tok_idx = sentence.find(token, next_char_idx)
            next_char_idx: int = tok_idx + len(token)
        found_tensors: List[Tensor] = []
        if tok_idx > -1:
            found_tensors = self.compute_embeddings(tok_idx, token, enc_output_no_batch, sentence, input_tensor,
                                                    tokenizer)
        if not len(found_tensors):
            raise Exception(f"{token} not found in input '{sentence}'")
        return found_tensors


def create_padding_mask(seq):
    seq = keras.backend.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
        keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return keras.backend.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type (padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = keras.backend.cast(keras.backend.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / keras.backend.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1.
    attention_weights = keras.backend.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

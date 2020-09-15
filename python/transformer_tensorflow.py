import numpy
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.text import Text
from scipy.spatial.distance import cosine
from tensorflow import keras, Tensor
from typing import List, Any, Tuple, Set
import tensorflow_datasets as tfds
from tensorflow.python.data import Dataset
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2
from tensorflow.python.ops.gen_math_ops import Mean
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from tqdm import tqdm

BUFFER_SIZE = 20000
BATCH_SIZE = 1000
MAX_LENGTH = 40
NUMBER_OF_LAYERS = 4
MODEL_DIMENSIONS = 128
FEED_FORWARD_DIMENSIONS = 512
NUMBER_OF_HEADS = 8
DROPOUT_RATE = 0.1
EPOCHS = 5


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

    def compute_embeddings(self, tok_idx: int, token: str, enc_output_no_batch: Tensor, sentence: str,
                           input_tensor: Tensor) -> List[Tensor]:
        found_tensors: List[Tensor] = []
        target_representation: List[int]
        next_idx: int = tok_idx + len(token)
        # add whitespace after the token if present in the input, otherwise use just the raw token
        if next_idx < len(sentence) and sentence[next_idx] == " ":
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
        found_tensors: List[Tensor] = []
        if tok_idx > -1:
            found_tensors = self.compute_embeddings(tok_idx, token, enc_output_no_batch, sentence, input_tensor)
        if not len(found_tensors):
            raise Exception(f"{token} not found in input '{sentence}'")
        return found_tensors


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)
    # Used in the 2nd attention block in the decoder. This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)
    # Used in the 1st attention block in the decoder. It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(keras.backend.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = keras.backend.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


def create_padding_mask(seq):
    seq = keras.backend.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions to add the padding to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def encode(lang1, lang2) -> Tuple[Tensor, Tensor]:
    lang1: Tensor = [tokenizer.vocab_size] + tokenizer.encode(lang1.numpy()) + [tokenizer.vocab_size + 1]
    lang2: Tensor = [tokenizer.vocab_size] + tokenizer.encode(lang2.numpy()) + [tokenizer.vocab_size + 1]
    return lang1, lang2


def evaluate(inp_sentence: str, tokenizer, transformer):
    start_token: List[int] = [tokenizer.vocab_size]
    end_token: List[int] = [tokenizer.vocab_size + 1]
    inp_sentence_enc: List[int] = start_token + tokenizer.encode(inp_sentence) + end_token
    encoder_input: Tensor = keras.backend.expand_dims(inp_sentence_enc, 0)
    decoder_input: List[int] = [tokenizer.vocab_size]
    output: Tensor = keras.backend.expand_dims(decoder_input, 0)
    attention_weights = None
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(
            [encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask])
        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = keras.backend.cast(keras.backend.argmax(predictions, axis=-1), tf.int32)
        # return the result if the predicted_id is equal to the end token
        if predicted_id == tokenizer.vocab_size + 1:
            return keras.backend.squeeze(output, axis=0), attention_weights
        # concatentate the predicted_id to the output which is given to the decoder as its input.
        output = keras.backend.concatenate([output, predicted_id], axis=-1)
    return keras.backend.squeeze(output, axis=0), attention_weights


def evaluate_lexis() -> None:
    examples: List[Tuple[str, str]] = [
        ("ego sum Alpha et Omega initium et finis", "ego sum Alpha et Omega principium et finis"),
        ("et secundus angelus tuba cecinit", "et septimus angelus tuba cecinit"),
        ("et ecce equus pallidus", "et ecce equus niger"),
        ("diliges proximum tuum sicut te ipsum", "diliges proximum tuum tamquam te ipsum"),
        ("si sic eum volo manere donec veniam quid ad te", "si sic eum volo manere donec venio quid ad te"),
        ("quaerebant ergo eum prendere", "quaerebant ergo eum adprehendere"),
        ("qui credit in me habet vitam aeternam", "qui credit in Filium habet vitam aeternam")
    ]
    sims: List[float] = []
    for example in examples:
        tensors_avg: List[Tensor] = []
        # for sent in example:
        for sent in [example[0], example[0]]:
            tokens: Set[str] = set(sent.split())
            tensors: List[Tensor] = []
            for tok in tokens:
                tensors += transformer.get_embeddings_for_token(sent, tokenizer, tok)
            tensors_avg.append(transformer.avg(tensors))
        cos_sim: float = 1 - cosine(tensors_avg[0], tensors_avg[1])
        sims.append(cos_sim)
    a = 0


def evaluate_polysemy(tokenizer: SubwordTextEncoder, transformer: Transformer):
    dataset_url: str = "https://box.hu-berlin.de/f/ce30d723beef4feba4c3/?dl=1"
    dataset_path: str = tf.keras.utils.get_file("pars_test.txt", dataset_url)
    # dataset_path: str = "../data/pars_test.txt"
    lines: List[str] = open(dataset_path).read().split("\n")
    examples: List[Example] = [Example(x) for x in lines if x]
    predictions: List[int] = []
    sims: List[float] = []
    for ex in examples:
        token1: str = ex.context1.content[ex.context1.token_range_start:ex.context1.token_range_end] + " "
        tensors1: List[Tensor] = transformer.get_embeddings_for_token(ex.context1.content, tokenizer, token1)
        token2: str = ex.context2.content[ex.context2.token_range_start:ex.context2.token_range_end] + " "
        tensors2: List[Tensor] = transformer.get_embeddings_for_token(ex.context2.content, tokenizer, token2)
        cos_sim: float = 1 - cosine(tensors1[0], tensors2[0])
        sims.append(cos_sim)
        predictions.append(1 if cos_sim > 0.4 else 0)
    print([x.label for x in examples])
    print(predictions)
    print(sims)
    correct_indices: List[int] = [i for i in range(len(predictions)) if predictions[i] == examples[i].label]
    print(correct_indices)
    print(f"Accuracy: {len(correct_indices) / len(examples) * 100}%")


def evaluate_polysemy_old(tokenizer: SubwordTextEncoder, transformer: Transformer):
    sentences: List[str] = [
        "et percussa est tertia pars solis et tertia pars lunae et tertia pars stellarum ut obscuraretur tertia pars eorum et diei non luceret pars tertia et nox similiter",
        "nam et pars quedam fluminis Nili ibi currit",
        "Ac saepe in eum locum ventum est tanto in omnes partes diviso equitatu ut modo visum ab se Ambiorigem in fuga circumspicerent captivi nec plane etiam abisse ex conspectu contenderent ut spe consequendi inlata atque infinito labore suscepto qui se summam a Caesare gratiam inituros putarent paene naturam studio vincerent semper que paulum ad summam felicitatem defuisse videretur atque ille latebris aut saltibus se eriperet et noctu occultatus alias regiones partes que peteret non maiore equitum praesidio quam quattuor quibus solis vitam suam committere audebat",
        "numquam ante arbitror te epistulam meam legisse nisi mea manu scriptam",
        "ante diem xii Kal Decembr Milo ante mediam noctem cum magna manu in campum venit",
        "numquam enim a Pomponia nostra certior sum factus esse cui dare litteras possem",
        "quod fere plerisque accidit ut praesidio litterarum diligentiam in perdiscendo ac memoriam remittant",
        "nam statim fidem publicam postulavit",
        "habete fidem Dei",
        "Fundamentum autem est iustitiae fides id est dictorum conventorum que constantia et veritas",
        "sol ",
        "merces "
    ]
    tokens: List[str] = ["pars", "pars", "partes", "manu", "manu", "litteras", "litterarum", "fidem", "fidem", "fides",
                         "sol", "merces"]
    # for tok in tokens:
    #     print(f"{tok}: {most_similar(transformer, tok)}")
    print_tokens: List[str] = [
        "pars solis", "pars fluminis", "equitatus in omnes partes divisus", "manu scriptus", "magna manus",
        "litteras dare alicui", "praesidium litterarum", "fides publica", "fides dei", "fides iustitiae", "sol",
        "merces"]
    sims: numpy.ndarray = numpy.zeros((len(tokens), len(tokens)))
    cross_validation_k: int = 5
    for k in range(cross_validation_k):
        relevant_tensors: List[Tensor] = []
        for i in range(len(sentences)):
            tensors: List[Tensor] = transformer.get_embeddings_for_token(sentences[i], tokenizer, tokens[i])
            relevant_tensors.append(tensors[0])
            for j in range(len(relevant_tensors) - 1):
                cos_sim: float = 1 - cosine(relevant_tensors[-1], relevant_tensors[j])
                sims[i, j] = sims[j, i] = round((sims[i, j] + cos_sim) / 2, 2) if sims[i, j] else cos_sim
    plot_similarities(print_tokens, sims)


def evaluate_word_order():
    dataset_path: str = "../data/word_order_test.txt"
    lines: List[str] = open(dataset_path).read().split("\n")
    examples: List[Example] = [Example(x) for x in lines if x]
    sims: List[Tuple[float, float]] = []
    for ex in examples:
        token1: str = ex.context1.content[ex.context1.token_range_start:ex.context1.token_range_end]
        tensors1: List[Tensor] = transformer.get_embeddings_for_token(ex.context1.content, tokenizer, token1)
        token2: str = ex.context2.content[ex.context2.token_range_start:ex.context2.token_range_end]
        tensors2: List[Tensor] = transformer.get_embeddings_for_token(ex.context2.content, tokenizer, token2)
        cos_sim: float = 1 - cosine(tensors1[0], tensors2[0])
        cos_sim2: float = 1 - cosine(tensors1[0], tensors1[0])
        sims.append((cos_sim, cos_sim2))
    a = 0


def filter_max_length(x, y, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)


def find_word_senses(tokenizer: SubwordTextEncoder, transformer: Transformer, dataset_path: str) -> None:
    word_forms_set: Set[str] = {"pars", "partis", "parti", "partem", "parte", "partes", "partium", "partibus"}
    examples: List[str] = open(dataset_path).read().split("\n")
    examples = [y for x in examples for y in x.split("\t")]
    example_token_sets: List[Set[str]] = [set(x.split()) for x in examples]
    deletion_indices: List[int] = [i for i in range(len(examples)) if
                                   not len(word_forms_set.intersection(example_token_sets[i]))]
    examples = [examples[i] for i in range(len(examples)) if i not in deletion_indices]
    examples_set: Set[str] = set(examples)
    relevant_tensors: List[Tensor] = []
    for example in tqdm(examples_set):
        target_token: str = next(x for x in example.split() if x in word_forms_set)
        tensors: List[Tensor] = transformer.get_embeddings_for_token(example, tokenizer, target_token)
        relevant_tensors.append(tensors[0])
    sims: numpy.ndarray = numpy.zeros((len(relevant_tensors), len(relevant_tensors)))
    for i in range(len(relevant_tensors)):
        for j in range(len(relevant_tensors) - 1):
            if i == j:
                continue
            cos_sim: float = 1 - cosine(relevant_tensors[i], relevant_tensors[j])
            sims[i, j] = sims[j, i] = round(cos_sim, 2)
    examples = [x[:20] for x in examples]
    sims_with_ex: List[Tuple[float, str, str]] = []
    for i in range(len(sims)):
        for j in range(len(sims[i])):
            sims_with_ex.append((sims[i, j], examples[i], examples[j]))
    sims_with_ex = [x for x in sims_with_ex if x[0]]
    sims_with_ex.sort(key=lambda x: x[0], reverse=True)
    sims_with_ex = sims_with_ex[:5] + sims_with_ex[-5:]
    for swe in sims_with_ex:
        print(swe)


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def loss_function(real, pred, loss_object):
    mask = tf.math.logical_not(keras.backend.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = keras.backend.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def plot_attention_weights(attention, sentence, result, layer, tokenizer):
    # attention, sentence, result, layer, tokenizer_pt, tokenizer_en
    fig = plt.figure(figsize=(16, 8))
    # sentence = tokenizer_pt.encode(sentence)
    sentence = tokenizer.encode(sentence)
    attention = keras.backend.squeeze(attention[layer], axis=0)
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)
        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))
        ax.set_ylim(len(result) - 1.5, -0.5)
        # ax.set_xticklabels(
        #     ['<start>'] + [tokenizer_pt.decode([i]) for i in sentence] + ['<end>'], fontdict=fontdict, rotation=90)
        ax.set_xticklabels(
            ['<start>'] + [tokenizer.decode([i]) for i in sentence] + ['<end>'], fontdict=fontdict, rotation=90)
        # ax.set_yticklabels([tokenizer_en.decode([i]) for i in result if i < tokenizer_en.vocab_size], fontdict=fontdict)
        ax.set_yticklabels([tokenizer.decode([i]) for i in result], fontdict=fontdict)
        ax.set_xlabel('Head {}'.format(head + 1))
    plt.tight_layout()
    plt.show()


def plot_similarities(print_tokens: List[str], sims: numpy.ndarray):
    ax: Axes
    fig: Figure
    fig, ax = plt.subplots()
    im: AxesImage = ax.imshow(sims)
    # We want to show all ticks...
    ax.set_xticks(np.arange(len(print_tokens)))
    ax.set_yticks(np.arange(len(print_tokens)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(print_tokens)
    ax.set_yticklabels(print_tokens)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(print_tokens)):
        for j in range(len(print_tokens)):
            value: str = str(sims[i, j])[2:]
            value += "0" if len(value) == 1 else ""
            text: Text = ax.text(j, i, value, ha="center", va="center", color="w")
    ax.set_title("Cosine similarity for various word senses")
    fig.tight_layout()
    plt.show()


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


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


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


def train_model(train_loss: Mean, train_accuracy: SparseCategoricalAccuracy, train_dataset: Dataset,
                ckpt_manager: CheckpointManager):
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch_idx, (input_tensor, target)) in enumerate(train_dataset):
            train_step(input_tensor, target)
            if batch_idx % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Batch {batch_idx} Loss {train_loss.result()} Accuracy {train_accuracy.result()}')
        if (epoch + 1) % 5 == 0:
            ckpt_save_path: str = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')
        print(f'Epoch {epoch + 1} Loss {train_loss.result()} Accuracy {train_accuracy.result()}')
        print(f'Time taken for 1 epoch: {time.time() - start} secs\n')


def tf_encode(la1: Tensor, la2: Tensor):  # pt, en
    # result_pt, result_en = tf.py_function(encode, [pt, en], [tf.int64, tf.int64])
    # result_pt.set_shape([None])
    # result_en.set_shape([None])
    # return result_pt, result_en
    result_la1, result_la2 = tf.py_function(encode, [la1, la2], [tf.int64, tf.int64])
    result_la1.set_shape([None])
    result_la2.set_shape([None])
    return result_la1, result_la2


# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp: Tensor, tar: Tensor):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer([inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask])
        loss = loss_function(tar_real, predictions, loss_object)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    train_loss(loss)
    train_accuracy(tar_real, predictions)


def translate(sentence: str, tokenizer, transformer, plot='') -> None:
    # sentence: str, tokenizer_pt, tokenizer_en, transformer, plot = ''
    # result, attention_weights = evaluate(sentence, tokenizer_pt, tokenizer_en, transformer)
    result, attention_weights = evaluate(sentence, tokenizer, transformer)
    # predicted_sentence = tokenizer_en.decode([i for i in result if i < tokenizer_en.vocab_size])
    predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])
    print(f'Input: {sentence}')
    print(f'Predicted translation: {predicted_sentence}')
    if plot:
        # plot_attention_weights(attention_weights, sentence, result, plot, tokenizer_pt, tokenizer_en)
        plot_attention_weights(attention_weights, sentence, result, plot, tokenizer)


def generate_examples(file_path: str) -> Tuple[str, str]:
    with open(file_path) as f:
        while True:
            line: str = f.readline()
            if not line:
                break
            line_parts: List[str] = line.split("\t")
            yield line_parts[0], line_parts[1]


def generate_train_examples() -> Tuple[str, str]:
    # train_dataset_fp: str = "../data/pars.txt"
    train_dataset_fp: str = tf.keras.utils.get_file("proiel.txt",
                                                    "https://box.hu-berlin.de/f/7da8d9c5703440e88531/?dl=1")
    # train_dataset_fp: str = tf.keras.utils.get_file("cc_train.txt",
    #                                                 "https://box.hu-berlin.de/f/f9a36dcb16e945b4a179/?dl=1")
    return generate_examples(train_dataset_fp)


def generate_val_examples() -> Tuple[str, str]:
    # val_dataset_fp: str = "./.data/cc_val.txt"
    val_dataset_fp: str = tf.keras.utils.get_file("cc_val.txt",
                                                  "https://box.hu-berlin.de/f/41a95d07b791433b919b/?dl=1")
    return generate_examples(val_dataset_fp)


def predict_next_sentence(sentence: str) -> None:
    result, attention_weights = evaluate(sentence, tokenizer, transformer)
    predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])
    print(f'Input: {sentence}')
    print(f'Predicted translation: {predicted_sentence}')


checkpoint_path: str = "./checkpoints/train"
train_examples: Dataset = tf.data.Dataset.from_generator(
    generate_train_examples, (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([]))).take(500)
tokenizer_path: str = "tokenizer.subwords"
tokenizer_prefix: str = tokenizer_path.split(".")[0]
tokenizer: SubwordTextEncoder
try:
    tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_prefix)
except NotFoundError:
    tokenizer: SubwordTextEncoder = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (la1.numpy() + la2.numpy()[:-1] for la1, la2 in train_examples), target_vocab_size=2 ** 13)
tokenizer.save_to_file(tokenizer_prefix)
train_dataset: Dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
val_examples: Dataset = tf.data.Dataset.from_generator(
    generate_val_examples, (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([]))).take(5000)
val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(BATCH_SIZE)
input_vocabulary_size = target_vocabulary_size = tokenizer.vocab_size + 2
# TODO: USE VALIDATION DATASET DURING TRAINING!
np.set_printoptions(suppress=True)
learning_rate: CustomSchedule = CustomSchedule(MODEL_DIMENSIONS)
optimizer: OptimizerV2 = keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
loss_object: SparseCategoricalCrossentropy = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
train_loss: Mean = keras.metrics.Mean(name='train_loss')
train_accuracy: SparseCategoricalAccuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
transformer: Transformer = Transformer(
    NUMBER_OF_LAYERS, MODEL_DIMENSIONS, NUMBER_OF_HEADS, FEED_FORWARD_DIMENSIONS, input_vocabulary_size,
    target_vocabulary_size, pe_input=input_vocabulary_size, pe_target=target_vocabulary_size, rate=DROPOUT_RATE)
ckpt: Checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
ckpt_manager: CheckpointManager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
evaluate_polysemy(tokenizer, transformer)
# evaluate_word_order()
# evaluate_lexis()
cluster_
# train_model(train_loss, train_accuracy, train_dataset, ckpt_manager)
evaluate_polysemy(tokenizer, transformer)
predict_next_sentence("Gallia est omnis divisa in partes tres.")
predict_next_sentence("Arma virumque cano Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora.")
predict_next_sentence(
    "Omnis homines qui sese student praestare ceteris animalibus summa ope niti decet ne vitam silentio transeant veluti pecora quae natura prona atque ventri oboedientia finxit.")

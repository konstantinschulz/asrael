import os
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.text import Text
from scipy.spatial.distance import cosine
from tensorflow import keras, Tensor
from typing import List, Tuple, Set
import tensorflow_datasets as tfds
from tensorflow.python.data import Dataset
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.keras.metrics import SparseCategoricalAccuracy
from tensorflow.python.ops.gen_math_ops import Mean
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.util import Checkpoint
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from tqdm import tqdm

from classes import Cluster, create_padding_mask, Transformer, Example, scaled_dot_product_attention, CustomSchedule

BUFFER_SIZE = 20000
BATCH_SIZE = 1000
MAX_LENGTH = 40
NUMBER_OF_LAYERS = 4
MODEL_DIMENSIONS = 128
FEED_FORWARD_DIMENSIONS = 512
NUMBER_OF_HEADS = 8
DROPOUT_RATE = 0.1
EPOCHS = 5

tokenizer: SubwordTextEncoder
transformer: Transformer


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
        token1: str = ex.context1.content[ex.context1.token_range_start:ex.context1.token_range_end]
        tensors1: List[Tensor] = transformer.get_embeddings_for_token(ex.context1.content, tokenizer, token1)
        token2: str = ex.context2.content[ex.context2.token_range_start:ex.context2.token_range_end]
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
    sims: np.ndarray = np.zeros((len(tokens), len(tokens)))
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
    sims: np.ndarray = np.zeros((len(relevant_tensors), len(relevant_tensors)))
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


def plot_similarities(print_tokens: List[str], sims: np.ndarray):
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


def print_out(q, k, v):
    temp_out, temp_attn = scaled_dot_product_attention(q, k, v, None)
    print('Attention weights are:')
    print(temp_attn)
    print('Output is:')
    print(temp_out)


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
        loss = loss_function(tar_real, predictions, transformer.loss_object)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    transformer.optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    transformer.train_loss(loss)
    transformer.train_accuracy(tar_real, predictions)


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


def predict_next_sentence(sentence: str, tokenizer: SubwordTextEncoder, transformer: Transformer) -> None:
    result, attention_weights = evaluate(sentence, tokenizer, transformer)
    predicted_sentence = tokenizer.decode([i for i in result if i < tokenizer.vocab_size])
    print(f'Input: {sentence}')
    print(f'Predicted translation: {predicted_sentence}')


def do_deep_learning():
    checkpoint_path: str = "./checkpoints/train"
    train_examples: Dataset = tf.data.Dataset.from_generator(
        generate_train_examples, (tf.string, tf.string), (tf.TensorShape([]), tf.TensorShape([]))).take(500)
    tokenizer_path: str = "tokenizer.subwords"
    tokenizer_prefix: str = tokenizer_path.split(".")[0]
    tokenizer: SubwordTextEncoder
    try:
        tokenizer = SubwordTextEncoder.load_from_file(tokenizer_prefix)
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
    transformer: Transformer = Transformer(
        NUMBER_OF_LAYERS, MODEL_DIMENSIONS, NUMBER_OF_HEADS, FEED_FORWARD_DIMENSIONS, input_vocabulary_size,
        target_vocabulary_size, pe_input=input_vocabulary_size, pe_target=target_vocabulary_size, rate=DROPOUT_RATE)
    ckpt: Checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=transformer.optimizer)
    ckpt_manager: CheckpointManager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    evaluate_polysemy(tokenizer, transformer)
    # evaluate_word_order()
    # evaluate_lexis()
    data_dir: str = "../data"
    proiel_pickle_path: str = os.path.join(data_dir, "proiel_conllu.pickle")
    cache_path: str = os.path.join(data_dir, "sense_embeddings.json")
    # build_sense_embeddings(proiel_pickle_path, tokenizer, cache_path)
    # train_model(transformer.train_loss, transformer.train_accuracy, train_dataset, ckpt_manager)
    evaluate_polysemy(tokenizer, transformer)
    predict_next_sentence("Gallia est omnis divisa in partes tres.", tokenizer, transformer)
    predict_next_sentence("Arma virumque cano Troiae qui primus ab oris Italiam fato profugus Laviniaque venit litora.",
                          tokenizer, transformer)
    predict_next_sentence(
        "Omnis homines qui sese student praestare ceteris animalibus summa ope niti decet ne vitam silentio transeant veluti pecora quae natura prona atque ventri oboedientia finxit.",
        tokenizer, transformer)


# do_deep_learning()

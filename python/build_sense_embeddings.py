import math
import random
from collections import Counter
from itertools import combinations

import numpy
import ujson
import multiprocessing
import os
import pickle
from typing import Dict, List, Set, Tuple
from conllu import TokenList
from matplotlib import pyplot
from numpy.polynomial import Polynomial
from scipy.spatial.distance import cosine
from tqdm import tqdm


def build_sense_embeddings(proiel_pickle: str, cache_dir: str):  # SubwordTextEncoder
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    lemma_dict: Dict[str, list] = dict()  # List[Cluster]
    input_queue: multiprocessing.Queue = multiprocessing.Queue()
    output_queue: multiprocessing.Queue = multiprocessing.Queue()

    def do_work(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
        from classes import Transformer
        from transformer_tensorflow import NUMBER_OF_LAYERS
        from transformer_tensorflow import MODEL_DIMENSIONS
        from transformer_tensorflow import NUMBER_OF_HEADS
        from transformer_tensorflow import FEED_FORWARD_DIMENSIONS
        from transformer_tensorflow import DROPOUT_RATE
        from tensorflow.python.training.tracking.util import Checkpoint
        import tensorflow as tf
        from tensorflow_datasets.core.features.text import SubwordTextEncoder
        from classes import Cluster
        from tensorflow.python.training.checkpoint_management import CheckpointManager
        tokenizer_path: str = "tokenizer.subwords"
        tokenizer_prefix: str = tokenizer_path.split(".")[0]
        tokenizer: SubwordTextEncoder
        tokenizer = SubwordTextEncoder.load_from_file(tokenizer_prefix)
        input_vocabulary_size = target_vocabulary_size = tokenizer.vocab_size + 2
        local_transformer: Transformer = Transformer(
            NUMBER_OF_LAYERS, MODEL_DIMENSIONS, NUMBER_OF_HEADS, FEED_FORWARD_DIMENSIONS, input_vocabulary_size,
            target_vocabulary_size, pe_input=input_vocabulary_size, pe_target=target_vocabulary_size, rate=DROPOUT_RATE)
        ckpt: Checkpoint = tf.train.Checkpoint(transformer=local_transformer, optimizer=local_transformer.optimizer)
        ckpt_manager: CheckpointManager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
        # if a checkpoint exists, restore the latest checkpoint.
        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')
        while True:
            sent: TokenList = input_queue.get(True)
            if sent is None:
                break
            lemma_dict: Dict[str, List[Cluster]] = process_sentence(sent, local_transformer, tokenizer)
            output_queue.put(lemma_dict)

    def process_sentence(sent: TokenList, local_transformer, tokenizer) -> dict:  # Transformer Dict[str, List[Cluster]]
        from classes import Cluster
        lemma_dict: Dict[str, List[Cluster]] = dict()
        content: str = " ".join([x["form"] for x in sent.tokens])
        ignore_set: Set[str] = set()
        for tok in sent.tokens:
            form: str = tok["form"]
            if form in ignore_set:
                continue
            tensors: list = local_transformer.get_embeddings_for_token(content, tokenizer, form)  # List[Tensor]
            if len(tensors) > 1:
                # all tensors for this form have already been added in the first run
                ignore_set.add(form)
            lemma: str = tok["lemma"]
            if lemma not in lemma_dict:
                lemma_dict[lemma] = []
            for tensor in tensors:
                # TODO: add form, sentenceID and tokenID
                lemma_dict[lemma].append(Cluster(tensors=[tensor]))
        return lemma_dict

    all_files: List[str] = [x for x in os.listdir(cache_dir) if x.endswith(".json")]
    indices: List[int] = [int(x.split(".")[0]) for x in all_files] + [-1]
    highest_index: int = max(indices)
    for i in range(len(conllu_all)):
        if i <= highest_index:
            continue
        input_queue.put(conllu_all[i])
    num_workers: int = os.cpu_count() + 1
    for i in range(num_workers):
        input_queue.put(None)
    pool: multiprocessing.Pool() = multiprocessing.Pool(num_workers, do_work, (input_queue, output_queue))
    percentile: int = int(len(conllu_all) / 100)
    for i in tqdm(range(highest_index + 1, len(conllu_all))):
        lemma_dict_part: Dict[str, list] = output_queue.get(True)  # List[Cluster]
        for lemma in lemma_dict_part:
            lemma_dict[lemma] = lemma_dict.get(lemma, []) + lemma_dict_part[lemma]
        if i % percentile == 0:
            cache_path: str = os.path.join(cache_dir, f"{i}.json")
            ujson.dump({k: [x.to_json() for x in v] for k, v in lemma_dict.items()}, open(cache_path, "w+"),
                       ensure_ascii=False)
            lemma_dict = dict()
    cache_path: str = os.path.join(cache_dir, f"{len(conllu_all)}.json")
    ujson.dump({k: [x.to_json() for x in v] for k, v in lemma_dict.items()}, open(cache_path, "w+"),
               ensure_ascii=False)


def build_sense_inventory(cache_dir: str, sense_inventory_path: str):
    lemma_dict: Dict[str, list] = dict()
    for file in tqdm([x for x in os.listdir(cache_dir) if x.endswith(".json")]):
        path: str = os.path.join(cache_dir, file)
        lemma_dict_part: Dict[str, list] = ujson.load(open(path))
        for key in lemma_dict_part:
            lemma_dict[key] = lemma_dict.get(key, []) + lemma_dict_part[key]
    ujson.dump(lemma_dict, open(sense_inventory_path, "w+"), ensure_ascii=False)


def cluster_word_senses(sense_inventory_path: str, sense_avg_dist_path: str):
    lemma_dict_raw: Dict[str, list] = ujson.load(open(sense_inventory_path))
    print("Preparing clusters...")
    lemma_dict_raw = {k: v for k, v in lemma_dict_raw.items() if len(v) > 1}
    with multiprocessing.Pool(processes=os.cpu_count() + 1) as pool:
        multiple_results: List[multiprocessing.pool.ApplyResult] = \
            [pool.apply_async(get_avg_dist, item) for item in lemma_dict_raw.items()]
        results: List[Tuple[str, float]] = [res.get(timeout=10) for res in tqdm(multiple_results)]
        ujson.dump(results, open(sense_avg_dist_path, "w+"))
    # for lemma in tqdm(lemma_dict):
    #     # for cluster in lemma_dict[lemma]:
    #     #     cluster.average_tensor = transformer.avg(cluster.tensors)
    #     # averages: List[Tensor] = [x.average_tensor for x in lemma_dict[lemma]]
    #     unique_pairs: List[Tuple[Tensor, Tensor]] = [comb for comb in combinations(averages, 2)]
    #     distances: List[float] = [cosine(t1, t2) for t1, t2 in unique_pairs]
    #     # min_dist: float = min(distances)
    #     # target_pair_idx: int = next(i for i in range(len(distances)) if distances[i] == min_dist)
    #     # target_averages: Tuple[Tensor, Tensor] = unique_pairs[target_pair_idx]
    #     # equality = tf.equal(lemma_dict[lemma][0].tensors[0], target_averages[0])
    #     # src_cluster: Cluster = next(x for x in lemma_dict[lemma] if all(tf.equal(x.average_tensor, target_averages[0])))
    #     # tgt_cluster: Cluster = next(x for x in lemma_dict[lemma] if all(tf.equal(x.average_tensor, target_averages[1])))
    #     # tgt_cluster.tensors += src_cluster.tensors
    #     # lemma_dict[lemma].remove(src_cluster)
    a = 0


def get_avg_dist(key: str, clusters: list):
    averages: list = [x["tensors"][0] for x in clusters]
    averages = random.sample(averages, 100) if len(averages) > 100 else averages
    unique_pairs: List[tuple] = [comb for comb in combinations(averages, 2)]
    distances: List[float] = [cosine(t1, t2) for t1, t2 in unique_pairs]
    avg_dist: float = sum(distances) / len(distances)
    return key, avg_dist


def inspect_distances(sense_avg_dist_path: str, proiel_pickle_path: str, fig_path: str) -> None:
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle_path, "rb"))
    example_dict: Dict[str, List[str]] = dict()
    target_set: Set[str] = {"Augustofratensis", "Sychem", "Constantinopolis", "emundatio", "Corazain", "datum",
                            "dimicatio", "acclamo", "histrio", "Siculus"}
    for sent in conllu_all:
        lemmata_set: Set[str] = set([x["lemma"] for x in sent.tokens])
        overlap: Set[str] = lemmata_set.intersection(target_set)
        if overlap:
            target_lemma: str = overlap.pop()
            example: str = " ".join([x["form"] for x in sent.tokens])
            print(sent.metadata["sent_id"], example)
            example_dict[target_lemma] = example_dict.get(target_lemma, []) + [example]
    a = 0


def measure_burstiness_proiel(proiel_pickle_path: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle_path, "rb"))
    counter: Counter = Counter()
    for sent in tqdm(conllu_all):
        lemmata: List[str] = [tok["lemma"] for tok in sent.tokens]
        # ignore_set: Set[str] = set()
        # for i in range(len(lemmata)):
        #     if lemmata[i] in counter:
        #         counter.pop(lemmata[i])
        #         ignore_set.add(lemmata[i])
        # counter.update(lemmata[i] for i in range(len(lemmata)) if lemmata[i] not in ignore_set)
        counter.update(lemmata)
    for k, v in [x for x in counter.items()]:
        if v != 2:
            counter.pop(k)
    burst_set: Set[str] = set()
    for sent in conllu_all:
        lemmata: List[str] = [tok["lemma"] for tok in sent.tokens]
        for lemma in lemmata:
            if lemma in counter and lemmata.count(lemma) == 2:
                burst_set.add(lemma)
    a = 0


def plot_cosine_distance_by_frequency(sense_avg_dist_path: str, proiel_pickle_path: str, fig_path: str) -> None:
    dist_list: List[Tuple[str, float]] = ujson.load(open(sense_avg_dist_path))
    dist_list.sort(key=lambda x: x[1])
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle_path, "rb"))
    counter: Counter = Counter()
    for sent in tqdm(conllu_all):
        counter.update([tok["lemma"] for tok in sent.tokens])
    dist_lemma_list: List[Tuple[str, float, int]] = []
    for dist in dist_list:
        dist_lemma_list.append((dist[0], dist[1], counter[dist[0]]))
    dist_lemma_list.sort(key=lambda x: x[2])
    x_data: List[float] = [math.log(x[2]) for x in dist_lemma_list]
    y_data: List[float] = [x[1] for x in dist_lemma_list]
    pyplot.scatter(x_data, y_data)
    x_distinct: List[float] = list(sorted(set(x_data)))
    x_arr: numpy.ndarray = numpy.array(x_distinct)
    m, b = numpy.polyfit(x_data, y_data, 1)
    pyplot.plot(x_arr, m * x_arr + b, "-r")
    pyplot.xlabel("log-normalized frequency of occurrence")
    pyplot.ylabel("Average cosine distance between vectors of the same lexeme")
    pyplot.title("Correlation of cosine distance and usage frequency")
    pyplot.savefig(fig_path, dpi=600)
    pyplot.show()


def test_sense_embeddings():
    sentence: str = "cum enim hoc rectum et gloriosum putarem ex annuo sumptu qui mihi decretus esset me C Coelio quaestori relinquere annuum referre in aerarium ad HS †cIↃ† ingemuit nostra cohors omne illud putans distribui sibi oportere ut ego amicior invenirer Phrygum et Cilicum aerariis quam nostro"
    token: str = "HS"
    from classes import Transformer
    from transformer_tensorflow import NUMBER_OF_LAYERS
    from transformer_tensorflow import MODEL_DIMENSIONS
    from transformer_tensorflow import NUMBER_OF_HEADS
    from transformer_tensorflow import FEED_FORWARD_DIMENSIONS
    from transformer_tensorflow import DROPOUT_RATE
    from tensorflow.python.training.tracking.util import Checkpoint
    import tensorflow as tf
    from tensorflow_datasets.core.features.text import SubwordTextEncoder
    from tensorflow.python.training.checkpoint_management import CheckpointManager
    tokenizer_path: str = "tokenizer.subwords"
    tokenizer_prefix: str = tokenizer_path.split(".")[0]
    tokenizer: SubwordTextEncoder
    tokenizer = SubwordTextEncoder.load_from_file(tokenizer_prefix)
    input_vocabulary_size = target_vocabulary_size = tokenizer.vocab_size + 2
    local_transformer: Transformer = Transformer(
        NUMBER_OF_LAYERS, MODEL_DIMENSIONS, NUMBER_OF_HEADS, FEED_FORWARD_DIMENSIONS, input_vocabulary_size,
        target_vocabulary_size, pe_input=input_vocabulary_size, pe_target=target_vocabulary_size, rate=DROPOUT_RATE)
    ckpt: Checkpoint = tf.train.Checkpoint(transformer=local_transformer, optimizer=local_transformer.optimizer)
    ckpt_manager: CheckpointManager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')
    tensors: list = local_transformer.get_embeddings_for_token(sentence, tokenizer, token)  # List[Tensor]
    a = 0


checkpoint_path: str = "./checkpoints/train"
data_dir: str = "../data"
proiel_pickle_path: str = os.path.join(data_dir, "proiel_conllu.pickle")
cache_dir: str = os.path.join(data_dir, "sense_embeddings")
sense_inventory_path: str = os.path.join(data_dir, "sense_inventory.json")
sense_avg_dist_path: str = os.path.join(data_dir, "sense_avg_dist.json")
fig_path: str = os.path.join(data_dir, "plots", "cos_dist_vs_freq.png")
os.makedirs(cache_dir, exist_ok=True)
# build_sense_embeddings(proiel_pickle_path, cache_dir)
# test_sense_embeddings()
# build_sense_inventory(cache_dir, sense_inventory_path)
# cluster_word_senses(sense_inventory_path, sense_avg_dist_path)
# plot_cosine_distance_by_frequency(sense_avg_dist_path, proiel_pickle_path, fig_path)
measure_burstiness_proiel(proiel_pickle_path)
# inspect_distances(sense_avg_dist_path, proiel_pickle_path, fig_path)

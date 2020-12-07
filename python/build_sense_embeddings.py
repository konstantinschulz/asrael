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
        from classes import ClusterItem
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
            lemma_dict: Dict[str, List[ClusterItem]] = process_sentence(sent, local_transformer, tokenizer)
            output_queue.put(lemma_dict)

    def process_sentence(sent: TokenList, local_transformer, tokenizer) -> dict:  # Transformer Dict[str, List[Cluster]]
        from classes import ClusterItem
        lemma_dict: Dict[str, List[ClusterItem]] = dict()
        content: str = " ".join([x["form"] for x in sent.tokens])
        ignore_set: Set[str] = set()
        sent_id: str = sent.metadata["sent_id"]
        for tok in sent.tokens:
            form: str = tok["form"]
            if form in ignore_set:
                continue
            try:
                tensors: list = local_transformer.get_embeddings_for_token(content, tokenizer, form)  # List[Tensor]
            except:
                continue
            if len(tensors) > 1:
                # all tensors for this form have already been added in the first run
                ignore_set.add(form)
            lemma: str = tok["lemma"]
            if lemma not in lemma_dict:
                lemma_dict[lemma] = []
            tok_id: int = tok["id"]
            for tensor in tensors:
                lemma_dict[lemma].append(
                    ClusterItem(form=form, segment_id=sent_id, token_id=tok_id, lemma=lemma, tensor=tensor))
        return lemma_dict

    all_files: List[str] = [x for x in os.listdir(cache_dir) if x.endswith(".pickle")]
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
            cache_path: str = os.path.join(cache_dir, f"{i}.pickle")
            pickle.dump(lemma_dict, open(cache_path, "wb+"))
            lemma_dict = dict()
    pool.close()
    pool.join()
    cache_path: str = os.path.join(cache_dir, f"{len(conllu_all)}.pickle")
    pickle.dump(lemma_dict, open(cache_path, "wb+"))


def build_sense_inventory(cache_dir: str, sense_inventory_path: str):
    lemma_dict: Dict[str, list] = dict()
    for file in tqdm([x for x in os.listdir(cache_dir) if x.endswith(".pickle")]):
        path: str = os.path.join(cache_dir, file)
        lemma_dict_part: Dict[str, list] = pickle.load(open(path, "rb"))
        for key in lemma_dict_part:
            lemma_dict[key] = lemma_dict.get(key, []) + lemma_dict_part[key]
    pickle.dump(lemma_dict, open(sense_inventory_path, "wb+"))


def calculate_type_token_ratio_proiel(proiel_pickle_path: str, fig_path: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle_path, "rb"))
    counter: Counter = Counter()
    sent_to_lemma_set: Dict[str, Set[str]] = {}
    sent_to_lemma_list: Dict[str, List[str]] = {}
    sent_to_idx: Dict[str, int] = {}
    lemma_to_sent: Dict[str, Set[str]] = {}
    for i in tqdm(range(len(conllu_all))):
        lemmata: List[str] = [tok["form"] for tok in conllu_all[i].tokens]
        lemmata_set: Set[str] = set(lemmata)
        counter.update(lemmata)
        sent_id: str = conllu_all[i].metadata["sent_id"]
        sent_to_lemma_set[sent_id] = lemmata_set
        sent_to_lemma_list[sent_id] = lemmata
        sent_to_idx[sent_id] = i
        for lemma in lemmata_set:
            if lemma not in lemma_to_sent:
                lemma_to_sent[lemma] = set()
            lemma_to_sent[lemma].add(sent_id)
    for k, v in [x for x in counter.items()]:
        if v <= 10:
            counter.pop(k)
    lemma_to_lemma_count: Dict[str, Tuple[int, float]] = {}
    lemma_to_lemmata: Dict[str, List[str]] = {}
    for lemma in tqdm(counter):
        for sent_id in lemma_to_sent[lemma]:
            lemma_to_lemmata[lemma] = lemma_to_lemmata.get(lemma, []) + sent_to_lemma_list[sent_id]
    for lemma in lemma_to_lemmata:
        sample_size: int = 100
        if len(lemma_to_lemmata[lemma]) < sample_size:
            continue
        all_lemmata: List[str] = random.sample(lemma_to_lemmata[lemma], sample_size)
        lemma_to_lemma_count[lemma] = (counter[lemma], len(set(all_lemmata)) / len(all_lemmata))
    x: List[float] = []
    y: List[float] = []
    for val in lemma_to_lemma_count.values():
        x += [math.log(val[0])]
        y += [val[1]]
    pyplot.scatter(x, y, s=1)
    x_distinct: List[float] = list(sorted(set(x)))
    x_arr: numpy.ndarray = numpy.array(x_distinct)
    params: numpy.ndarray = numpy.polyfit(x, y, 1)
    pyplot.plot(x_arr, numpy.polyval(params, x_arr), "-r")
    pyplot.xlabel("Log-normalized (ln) Frequency of Occurrence")
    pyplot.ylabel("Type-Token Ratio")
    pyplot.title("Correlation of Type-Token Ratio and Usage Frequency")
    pyplot.savefig(fig_path, dpi=600)
    pyplot.show()
    a = 0


def cluster_word_senses(src_path: str, similarity_threshold: float, target_path: str) -> None:
    from classes import Cluster, SimilarityItem

    max_samples: int = 15

    def chunks(lst: list, n: int):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def get_similarities(clusters: list, is_subsequent_iteration: bool) -> list:
        import tensorflow as tf
        sims: List[SimilarityItem] = []
        # for performance
        if len(clusters) > max_samples:
            clusters = random.sample(clusters, max_samples)
        unique_pairs: List[Tuple[Cluster, Cluster]] = [comb for comb in combinations(clusters, 2)]
        for up in unique_pairs:
            cluster1: Cluster = up[0]
            cluster2: Cluster = up[1]
            # if both clusters have not recently been merged, skip them
            if is_subsequent_iteration and len(cluster1.cluster_items) + len(cluster2.cluster_items) == 2:
                continue
            idx1: int = clusters.index(cluster1)
            idx2: int = clusters.index(cluster2)
            similarity: float = -tf.keras.losses.cosine_similarity(cluster1.get_average_tensor(),
                                                                   cluster2.get_average_tensor()).numpy()
            sims.append(SimilarityItem(idx1, idx2, similarity))
        sims.sort(key=lambda x: x.similarity, reverse=True)
        return sims

    def do_work(input_queue: multiprocessing.Queue, output_queue: multiprocessing.Queue):
        while True:
            input_item: Tuple[List[Cluster], bool] = input_queue.get(True)
            if input_item is None:
                break
            similarities: List[SimilarityItem] = get_similarities(input_item[0], input_item[1])
            while len(similarities) > 1 and similarities[0].similarity > similarity_threshold:
                target_cluster: Cluster = input_item[0][similarities[0].index1]
                src_cluster: Cluster = input_item[0][similarities[0].index2]
                target_cluster.cluster_items += src_cluster.cluster_items
                target_cluster.average_tensor = None
                input_item[0][similarities[0].index1] = target_cluster
                input_item[0].remove(src_cluster)
                similarities = get_similarities(input_item[0], input_item[1])
            output_queue.put(input_item[0])

    input_queue: multiprocessing.Queue = multiprocessing.Queue()
    output_queue: multiprocessing.Queue = multiprocessing.Queue()
    print("Loading clusters...")
    clusters_old: List[Cluster] = pickle.load(open(src_path, "rb"))
    clusters_new: List[Cluster] = []
    num_workers: int = os.cpu_count() + 1
    pool: multiprocessing.Pool() = multiprocessing.Pool(num_workers, do_work, (input_queue, output_queue))
    is_subsequent_iteration: bool = False
    print("Distributing workload...")
    while len(clusters_new) < len(clusters_old):
        clusters_old = clusters_new if len(clusters_new) > 0 else clusters_old
        random.shuffle(clusters_old)
        clusters_new = []
        batch_size: int = min(max_samples, len(clusters_old))
        chunk_count: int = 0
        for chunk in chunks(clusters_old, batch_size):
            input_queue.put((chunk, is_subsequent_iteration))
            chunk_count += 1
        for i in tqdm(range(chunk_count)):
            output_item: List[Cluster] = output_queue.get(True)
            clusters_new += output_item
        print(f"Old: {len(clusters_old)} --> New: {len(clusters_new)}")
        is_subsequent_iteration = True
    for i in range(num_workers):
        input_queue.put(None)
    pool.close()
    pool.join()
    pickle.dump(clusters_new, open(target_path, "wb+"))


def get_avg_dist(key: str, clusters: list):
    averages: list = [x["tensors"][0] for x in clusters]
    averages = random.sample(averages, 100) if len(averages) > 100 else averages
    unique_pairs: List[tuple] = [comb for comb in combinations(averages, 2)]
    distances: List[float] = [cosine(t1, t2) for t1, t2 in unique_pairs]
    avg_dist: float = sum(distances) / len(distances)
    return key, avg_dist


def inspect_avg_distance_between_word_senses(sense_inventory_path: str, sense_avg_dist_path: str):
    lemma_dict_raw: Dict[str, list] = pickle.load(open(sense_inventory_path, "rb"))
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


def inspect_cluster_size(clustered_path: str, sense_inventory_path: str, fig_path: str):
    from classes import Cluster
    # cluster_dict_original: Dict[str, List[Cluster]] = pickle.load(open(sense_inventory_path, "rb"))
    # clusters_count_original: int = sum([len(y.cluster_items) for x in cluster_dict_original.values() for y in x])
    cluster_dict_raw: Dict[str, List[Cluster]] = pickle.load(open(clustered_path, "rb"))
    largest_cluster: Cluster = Cluster([])
    for lemma in tqdm(cluster_dict_raw):
        for cluster in cluster_dict_raw[lemma]:
            if len(cluster.cluster_items) > len(largest_cluster.cluster_items):
                largest_cluster = cluster
    print(f"Max cluster size for {largest_cluster.get_dominant_lemma()}: {len(largest_cluster.cluster_items)}")
    merged_clusters: List[Cluster] = [y for x in cluster_dict_raw.values() for y in x if len(y.cluster_items) > 1]
    merged_clusters.sort(key=lambda x: len(x.cluster_items), reverse=True)
    merged_cluster_items_count: int = sum([len(x.cluster_items) for x in merged_clusters])
    x_data: List[str] = list(range(len(merged_clusters) - 1))
    y_data: List[int] = [len(x.cluster_items) for x in merged_clusters[1:]]
    pyplot.scatter(x_data, y_data)
    pyplot.xlabel("Cluster ID")
    pyplot.ylabel("Number of contained usage contexts")
    pyplot.title("Distribution of cluster size")
    pyplot.savefig(fig_path, dpi=600)
    pyplot.show()


def inspect_clusters(clustered_path: str, proiel_pickle_path: str):
    from classes import Cluster
    clusters: List[Cluster] = pickle.load(open(clustered_path, "rb"))
    merged_clusters: List[Cluster] = [x for x in clusters if len(x.cluster_items) > 1]
    conllu: List[TokenList] = pickle.load(open(proiel_pickle_path, "rb"))
    segment_ids: Set[str] = set([y.segment_id for x in merged_clusters for y in x.cluster_items])
    sents: List[TokenList] = [x for x in conllu if x.metadata["sent_id"] in segment_ids]
    forms: List[List[str]] = [[y.form for y in x.cluster_items] for x in merged_clusters]
    merged_contexts_count: int = sum([1 for x in forms for y in x])
    unmerged_contexts_count: int = sum([1 for x in clusters if len(x.cluster_items) == 1])
    sent_content: List[str] = [" ".join([y["form"] for y in x.tokens]) for x in sents if
                               x.metadata["sent_id"] == "77314"]
    for cluster in tqdm(clusters):
        if len(cluster.cluster_items) > 1:
            lemmata: List[str] = [x.lemma for x in cluster.cluster_items]
            if len(set(lemmata)) > 1:
                print(set(lemmata))


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


def measure_burstiness_proiel(proiel_pickle_path: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle_path, "rb"))
    counter: Counter = Counter()
    for sent in tqdm(conllu_all):
        lemmata: List[str] = [tok["lemma"] for tok in sent.tokens]
        counter.update(lemmata)
    for k, v in [x for x in counter.items()]:
        if v <= 2:
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
    pyplot.scatter(x_data, y_data, 1)
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
sense_inventory_path: str = os.path.join(data_dir, "sense_inventory.pickle")
sense_avg_dist_path: str = os.path.join(data_dir, "sense_avg_dist.json")
fig_path: str = os.path.join(data_dir, "plots", "cos_dist_vs_freq.png")
fig_path_ttr_vs_freq: str = os.path.join(data_dir, "plots", "ttr_vs_freq.png")
fig_path_cluster_size_distribution: str = os.path.join(data_dir, "plots", "cluster_size_distribution.png")
clustering_0_9_path: str = os.path.join(data_dir, "sense_inventory_0_9.pickle")
clustering_0_8_path: str = os.path.join(data_dir, "sense_inventory_0_8.pickle")
clustering_0_7_path: str = os.path.join(data_dir, "sense_inventory_0_7.pickle")
os.makedirs(cache_dir, exist_ok=True)
# build_sense_embeddings(proiel_pickle_path, cache_dir)
# test_sense_embeddings()
# build_sense_inventory(cache_dir, sense_inventory_path)
# plot_cosine_distance_by_frequency(sense_avg_dist_path, proiel_pickle_path, fig_path)
# measure_burstiness_proiel(proiel_pickle_path)
# calculate_type_token_ratio_proiel(proiel_pickle_path, fig_path_ttr_vs_freq)
# inspect_distances(sense_avg_dist_path, proiel_pickle_path, fig_path)
# cluster_word_senses(clustering_0_9_path, 0.8, clustering_0_8_path)
# cluster_word_senses(clustering_0_8_path, 0.7, clustering_0_7_path)
# inspect_cluster_size(clustering_0_9_path, sense_inventory_path, fig_path_cluster_size_distribution)
inspect_clusters(clustering_0_8_path, proiel_pickle_path)

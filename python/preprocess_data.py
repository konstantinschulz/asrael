import os
import pickle
import random
from itertools import combinations
from typing import List, Dict, Set, Tuple

import conllu
import numpy
from conllu import TokenList
from numpy.core.multiarray import ndarray
from scipy.spatial.distance import cosine
from tqdm import tqdm


def calculate_contexts_dvs(proiel_pickle: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    lengths: Set[int] = set()
    voc_set: Set[str] = set()
    target_annotation: str = "lemma"
    print("Metadata...")
    for sent in tqdm(conllu_all):
        lengths.add(len(sent.tokens))
        for tok in sent.tokens:
            voc_set.add(tok[target_annotation])
    max_len: int = max(lengths)
    voc: Dict[str, int] = {}
    for entry in voc_set:
        voc[entry] = len(voc) + 1
    vec_dict: Dict[str, List[ndarray]] = dict()
    print("Vectorizing...")
    for sent in tqdm(conllu_all):
        vec: ndarray = numpy.zeros(max_len)
        tokens: List[str] = [x[target_annotation] for x in sent.tokens]
        for i in range(len(tokens)):
            vec[i] = voc[tokens[i]]
        for j in range(len(tokens)):
            vec_dict[tokens[j]] = vec_dict.get(tokens[j], []) + [vec]
    avg_dict: Dict[str, float] = dict()
    print("Averaging...")
    for entry in tqdm(vec_dict):
        vec_list: List[ndarray] = vec_dict[entry]
        vec_list_len: int = len(vec_list)
        if vec_list_len < 10:
            continue
        elif vec_list_len > 100:
            vec_list = random.sample(vec_list, 100)
        pairs: List[Tuple[ndarray, ndarray]] = [comb for comb in combinations(vec_list, 2)]
        max_pairs_len: int = 100
        if len(pairs) > max_pairs_len:
            pairs = random.sample(pairs, max_pairs_len)
        sims: List[float] = [1 - cosine(x[0], x[1]) for x in pairs]
        if len(sims):
            avg_dict[entry] = sum(sims) / len(sims)
    avg_sorted: list = sorted(avg_dict.items(), key=lambda x: x[1], reverse=False)
    top10_sorted: List[str] = [x[0] for x in (avg_sorted[:5] + avg_sorted[-5:])]
    lemma_dict: Dict[str, Tuple[float, int, int]] = dict()
    for lemma in top10_sorted:
        example_contexts: List[str] = [" ".join([z["form"] for z in x.tokens]) for x in conllu_all if
                                       any(y for y in x.tokens if y["lemma"] == lemma)]
        context_set: Set[str] = set()
        for context in example_contexts:
            for tok in context.split():
                context_set.add(tok)
        tok_count: int = sum([len(x.split()) for x in example_contexts])
        type_count: int = len(context_set)
        lemma_dict[lemma] = (type_count / tok_count, type_count, tok_count)
    a = 0


def calculate_contexts_raw(proiel_pickle: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    cooccurrence_dict: Dict[str, Set[str]] = dict()
    for sent in tqdm(conllu_all):
        tokens: List[str] = [x["form"] for x in sent.tokens]
        for i in range(len(tokens)):
            if tokens[i] not in cooccurrence_dict:
                cooccurrence_dict[tokens[i]] = set()
            for j in range(len(tokens)):
                if j != i:
                    cooccurrence_dict[tokens[i]].add(tokens[j])
    cooccurrences_sorted: list = sorted(cooccurrence_dict.items(), key=lambda x: len(x[1]), reverse=True)
    a = 0


def calculate_types_per_lemma(proiel_pickle: str, french_path: str, german_path: str) -> None:
    # conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    # print("Loading French...")
    # french_tb: List[TokenList] = conllu.parse(open(french_path).read())
    print("Loading German...")
    german_tb: List[TokenList] = conllu.parse(open(german_path).read())
    for tb in tqdm([german_tb]):  # conllu_all, french_tb,
        lemma_dict: Dict[str, Set[str]] = dict()
        for sent in tb:
            for tok in sent.tokens:
                lemma: str = tok["lemma"]
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = set()
                lemma_dict[lemma].add(tok["form"])
        lemma_count: int = len(lemma_dict)
        type_count: int = sum([len(x) for x in lemma_dict.values()])
        print(f"Average types per lemma: {type_count / lemma_count}")


def calculate_verb_to_noun_ratio(conllu_all: List[TokenList]) -> float:
    noun_count: int = 0
    verb_count: int = 0
    for sent in conllu_all:
        for tok in sent.tokens:
            pos_tag: str = tok["upos"]
            if pos_tag == "VERB":
                verb_count += 1
            elif pos_tag == "NOUN":
                noun_count += 1
    # formula: noun_count / (noun_count + verb_count)
    return noun_count / (noun_count + verb_count)


def calculate_verb_to_noun_ratio_proiel(proiel_pickle: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    # pos_tags: Set[str] = set([y["upos"] for x in conllu_all for y in x.tokens])
    # noun-verb-ratio: 0.50
    a = calculate_verb_to_noun_ratio(conllu_all)
    b = 0


def calculate_verb_to_noun_ratio_ud(french_path: str, german_path: str):
    # conllu_all: List[TokenList] = conllu.parse(open(french_path).read())
    # french_ratio = calculate_verb_to_noun_ratio(conllu_all)  # 0.71
    conllu_all = conllu.parse(open(german_path).read())
    german_ratio = calculate_verb_to_noun_ratio(conllu_all)  # 0.73
    b = 0


def create_dataset_cc(raw_dataset_path: str) -> None:
    train_dataset_path: str = "./.data/cc_train.txt"
    val_dataset_path: str = "./.data/cc_val.txt"
    line_number: int = 0
    buffer: List[str] = []
    with open(train_dataset_path, "w+") as train_file:
        with open(val_dataset_path, "w+") as val_file:
            for line in open(raw_dataset_path).readlines():
                line_number += 1
                buffer.append(line[:-1].replace("\t", " "))
                if line_number % 2 == 0:
                    target_file = val_file if line_number % 20 == 0 else train_file
                    target_file.write("\t".join(buffer) + "\n")
                    buffer = []
                    if line_number % 100_000 == 0:
                        print(line_number)


def create_dataset_proiel(proiel_pickle: str) -> None:
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    dataset_path: str = "../data/proiel.txt"
    sentence_count: int = len(conllu_all)
    with open(dataset_path, "w+") as f:
        for i in range(sentence_count):
            if i % 2 == 0 and i < sentence_count - 1:
                sentences: List[str] = []
                for j in [i, i + 1]:
                    sentences.append(" ".join([x["form"] for x in conllu_all[j].tokens]))
                f.write("\t".join(sentences) + "\n")
    # frequencies: Dict[str, int] = {}
    # target_lemmata: Set[str] = {"littera", "fides", "finis"}  # {"gratia", "pars", "causa"}
    # examples: Dict[str, List[str]] = {}
    # for sent in conllu_all:
    #     for tok in sent.tokens:
    #         lemma: str = tok["lemma"]
    #         frequencies[lemma] = frequencies.get(lemma, 0) + 1
    #         if lemma in target_lemmata:
    #             examples[lemma] = examples.get(lemma, []) + [" ".join([x["form"] for x in sent.tokens])]
    # frequencies_sorted: list = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)  # List[Tuple[str, int]]
    # a = 0


def create_dataset_pars() -> None:
    source_path: str = "../data/pars.txt"
    examples_list: List[str] = open(source_path).read().split("\n")
    with open("pars.txt", "w+") as f:
        for i in range(len(examples_list)):
            if i + 1 == len(examples_list):
                break
            if i % 2 != 0:
                f.write("\t".join([examples_list[i], examples_list[i + 1]]) + "\n")


def find_lexically_identical_sentences(proiel_pickle: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    sentences: List[Set[str]] = []
    for sent in conllu_all:
        sentences.append(set([x["form"] for x in sent.tokens]))
    for i in tqdm(range(len(sentences))):
        for j in range(len(sentences)):
            if sentences[j] == sentences[i] and i != j:
                sentence1: List[str] = [x["form"] for x in conllu_all[i].tokens]
                sentence2: List[str] = [x["form"] for x in conllu_all[j].tokens]
                if sentence1 != sentence2:
                    print(i, j, " ".join(sentence1), " ".join(sentence2))


def find_syntactically_identical_sentences(proiel_pickle: str):
    conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
    sentences: List[Set[str]] = []
    for sent in conllu_all:
        sentences.append(set([x["form"] for x in sent.tokens]))
    for i in tqdm(range(len(sentences))):
        for j in range(len(sentences)):
            intersection: Set[str] = sentences[j].intersection(sentences[i])
            if len(intersection) == len(sentences[i]) - 1 and i != j:
                sentence1: List[str] = [x["form"] for x in conllu_all[i].tokens]
                sentence2: List[str] = [x["form"] for x in conllu_all[j].tokens]
                if len(sentence1) == len(sentence2) > 3:
                    diff: List[int] = [i for i in range(len(sentence1)) if sentence1[i] != sentence2[i]]
                    if len(diff) == 1:
                        ids: List[str] = [conllu_all[k].metadata["sent_id"] for k in [i, j]]
                        print(ids, " ".join(sentence1), " ".join(sentence2))


data_dir: str = "../data"
raw_dataset_path: str = os.path.join(data_dir, "corpus_corporum", "corpus_corporum_tokenized.tsv")
proiel_pickle: str = os.path.join(data_dir, "proiel_conllu.pickle")
ud_path: str = os.path.join(data_dir, "universal_dependencies")
french_path: str = os.path.join(ud_path, "fr_ftb-ud-train.conllu")
german_path: str = os.path.join(ud_path, "de_hdt-ud-dev.conllu")
# create_dataset_proiel(proiel_pickle)
# create_dataset_pars()
# calculate_verb_to_noun_ratio_proiel(proiel_pickle)
# calculate_verb_to_noun_ratio_ud(french_path, german_path)
# calculate_contexts_raw(proiel_pickle)
# calculate_contexts_dvs(proiel_pickle)
# calculate_types_per_lemma(proiel_pickle, french_path, german_path)
# find_lexically_identical_sentences(proiel_pickle)
# find_syntactically_identical_sentences(proiel_pickle)

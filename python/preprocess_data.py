import json
import multiprocessing
import os
import pickle
import random
import sys
from itertools import combinations
from typing import List, Dict, Set, Tuple

import conllu
import numpy
import numpy as np
from conllu import TokenList
from conllu.models import Token
from graphannis.cs import CorpusStorageManager, ResultOrder, ImportFormat
from networkx import MultiDiGraph
from numpy.core.multiarray import ndarray
from scipy.spatial.distance import cosine
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, Row
from sqlalchemy.orm import Session
from tqdm import tqdm

from multiproc_classes import SentencePair, SentenceRelation, Base, Exclusion
from python.config import Config

cache_dir: str = os.path.abspath(".cache")


def calculate_contexts_dvs(proiel_pickle: str):
	""" Calculates the type / token ratio for selected example contexts of all lemmata in the corpus. """
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
		unique_pairs: List[Tuple[ndarray, ndarray]] = [comb for comb in combinations(vec_list, 2)]
		max_pairs_len: int = 100
		if len(unique_pairs) > max_pairs_len:
			unique_pairs = random.sample(unique_pairs, max_pairs_len)
		sims: List[float] = [1 - cosine(x[0], x[1]) for x in unique_pairs]
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
	""" Counts all co-occurrences for each token in PROIEL. """
	conllu_all: list[TokenList] = pickle.load(open(proiel_pickle, "rb"))
	cooccurrence_dict: dict[str, dict[str, int]] = dict()
	for sent in tqdm(conllu_all):
		tokens: list[str] = [x["form"] for x in sent.tokens]
		for i in range(len(tokens)):
			if tokens[i] not in cooccurrence_dict:
				cooccurrence_dict[tokens[i]] = dict()
			for j in range(len(tokens)):
				if j != i:
					cooccurrence_dict[tokens[i]][tokens[j]] = cooccurrence_dict[tokens[i]].get(tokens[j], 0) + 1
	# cooccurrences_sorted: list = sorted(cooccurrence_dict.items(), key=lambda x: len(x[1]), reverse=True)
	cooccurrences_sorted: dict[str, list[tuple[str, int]]] = dict()
	for key in cooccurrence_dict:
		cooccurrences_sorted[key] = sorted(cooccurrence_dict[key].items(), key=lambda x: x[1], reverse=True)
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


def check_sentence_pair_lexical_substitution(inputs: SentencePair):
	def check_sentences(srs: tuple[SentenceRelation, SentenceRelation]) -> bool:
		# the number of tokens has to be identical
		if len(srs[0].token_list) != len(srs[1].token_list):
			return False
		intersection: set[str] = srs[1].token_set.intersection(srs[0].token_set)
		# check if all types are identical except 1
		if len(intersection) != len(srs[0].token_set) - 1:
			return False
		# remove pairs where both sentences are very short
		if len(srs[0].token_list) == len(srs[1].token_list) < 4:
			return False
		sent_1: list[str] = srs[0].token_list
		diff: list[int] = [k for k in range(len(sent_1)) if sent_1[k] != srs[1].token_list[k]]
		# check if more than one token is different
		if len(diff) != 1:
			return False

	if check_sentences((inputs.sr_1, inputs.sr_2)):
		inputs.is_match = True
		print(inputs.sr_1.__dict__, inputs.sr_2.__dict__)
	return inputs


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


def create_dataset_pars() -> None:
	source_path: str = "../data/pars.txt"
	examples_list: List[str] = open(source_path).read().split("\n")
	with open("pars.txt", "w+") as f:
		for i in range(len(examples_list)):
			if i + 1 == len(examples_list):
				break
			if i % 2 != 0:
				f.write("\t".join([examples_list[i], examples_list[i + 1]]) + "\n")


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


def create_proiel_cache(proiel_dir: str, proiel_pickle: str) -> None:
	if os.path.exists(proiel_pickle):
		return
	conll: list[TokenList] = []
	for file in tqdm([x for x in os.listdir(proiel_dir) if x.endswith(".conllu")]):
		conll += conllu.parse(open(os.path.join(proiel_dir, file)).read())
	pickle.dump(conll, open(proiel_pickle, "wb+"))


def find_example_contexts_proiel():
	conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
	frequencies: Dict[str, int] = {}
	target_lemmata: Set[str] = {"littera", "fides", "finis"}  # {"gratia", "pars", "causa"}
	examples: Dict[str, List[str]] = {}
	for sent in conllu_all:
		for tok in sent.tokens:
			lemma: str = tok["lemma"]
			frequencies[lemma] = frequencies.get(lemma, 0) + 1
			if lemma in target_lemmata:
				examples[lemma] = examples.get(lemma, []) + [" ".join([x["form"] for x in sent.tokens])]
	frequencies_sorted: list = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)  # List[Tuple[str, int]]
	a = 0


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


def find_syntactically_identical_sentences(proiel_pickle: str, database_url: str):
	conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))
	engine: Engine = create_engine(database_url, future=True)
	Base.metadata.create_all(engine)
	random.shuffle(conllu_all)
	sr_dict: Dict[int, SentenceRelation] = dict()
	for i in range(len(conllu_all)):
		tokens: List[str] = [x["form"] for x in conllu_all[i].tokens]
		sent_id: str = conllu_all[i].metadata["sent_id"]
		sr_dict[i] = SentenceRelation(sent_id, " ".join(tokens), tokens, set(tokens))
	input_list: list[SentencePair] = []
	outputs: list[SentencePair] = []
	sr_cache: list[tuple[SentenceRelation, SentenceRelation]] = []
	dataset_size: int = len(sr_dict)
	num_workers: int = os.cpu_count() + 1
	threshold: int = max(num_workers * 10000, int(pow(dataset_size, 2) / 10000))
	pool: multiprocessing.Pool() = multiprocessing.Pool(num_workers)
	with Session(engine) as session:
		for i in tqdm(range(dataset_size)):
			for j in range(dataset_size):
				sr_cache.append((sr_dict[i], sr_dict[j]))
				if len(sr_cache) == threshold or i == j == dataset_size - 1:
					sentence_ids: list[tuple[str, str]] = list(map(lambda x: (x[0].id, x[1].id), sr_cache))
					pair_ids: list[str] = pool.starmap(get_sentence_pair_id, sentence_ids)
					# TODO: distribute the search to the workers via Queue
					sql: str = f"SELECT id FROM exclusion WHERE id in {tuple(pair_ids)}"
					exclusions: list[Row] = session.execute(sql).all()
					exclusion_set: set[str] = set([x.id for x in exclusions])
					for k in range(len(sr_cache)):
						if pair_ids[k] in exclusion_set:
							continue
						sp: SentencePair = SentencePair(sr_cache[k][0], sr_cache[k][1], pair_ids[k])
						input_list.append(sp)
					sr_cache = []
				if len(input_list) >= threshold or i == j == dataset_size - 1:
					outputs_new: list[SentencePair] = pool.map(check_sentence_pair_lexical_substitution, input_list)
					input_list = []
					new_exclusions: set[str] = set()
					output_size: int = len(outputs_new)
					for k in range(output_size):
						if outputs_new[k].is_match:
							outputs.append(outputs_new[k])
						else:
							new_exclusions.add(outputs_new[k].id)
					session.execute(Exclusion.__table__.insert(),
					                [{"id": exclusion_id} for exclusion_id in new_exclusions])
					session.commit()
	engine.dispose()
	pool.close()
	pool.join()
	json.dump([(x.sr_1.__dict__, x.sr_2.__dict__) for x in outputs], open("lexical_substitution_list.json", "w+"))


def find_syntactically_identical_sentences_annis():
	GRAPH_DATABASE_DIR: str = os.path.join(os.sep, "tmp", "graphannis-data")
	csm: CorpusStorageManager = CorpusStorageManager(GRAPH_DATABASE_DIR)
	corpus_name: str = "proiel"
	csm.import_from_fs(path="/home/konstantin/Applications/pepper/output", fmt=ImportFormat.RelANNIS,
	                   corpus_name=corpus_name, overwrite_existing=False)
	# aql: str = 'cat="S" & cat="S" & tok & tok & tok & tok & tok & tok & tok & tok & #1 _l_ #3 & #2 _l_ #4 & #3 == #4 ' \
	#            '& #1 _r_ #5 & #2 _r_ #6 & #5 == #6 & #3 . #7 & #4 . #8 & #7 == #8 & #9 . #5 & #10 . #6 & #9 == #10 & ' \
	#            '#1 .* #2'
	aql: str = 'cat="S" & tok & tok & #1 _i_ #2 & #1 _i_ #3 & #2 == #3'
	results: List[List[str]] = csm.find(corpus_name=corpus_name, query=aql, limit=sys.maxsize,
	                                    order=ResultOrder.NotSorted)
	span_set: Set[str] = set([y for x in results for y in x if "Span" in y])
	long_spans: Set[str] = set()
	for span in span_set:
		# remove namespace from the span ID
		span_no_ns: str = span.split("::")[-1]
		sg: MultiDiGraph = csm.subgraph(corpus_name, [span_no_ns])
		# get number of tokens in span; 1 node is an integer ID, the other nodes correspond to tokens
		if len(sg.nodes) > 3:
			long_spans.add(span_no_ns)
	a = 0


def find_syntactically_identical_sentences_with_nouns(proiel_pickle: str, similar_syntax_of_nouns_path: str):
	def process_sentence() -> dict[str, list[str]]:
		sim_dict: dict[str, list[str]] = dict()
		# the two sentences differ in length or are identical
		if sent_lengths[i] != sent_lengths[j] or i == j:
			return sim_dict
		# combined: list[tuple[str, str]] = list(zip(tok_lists[i], tok_lists[j]))
		diff: set[int] = set()
		diff_counter: int = 0
		for k in range(sent_lengths[i]):
			if tok_lists[i][k] != tok_lists[j][k]:  # combined[k][0] != combined[k][1]:
				diff_counter += 1
				if diff_counter == 2:
					break
				diff.add(k)
		# the two sentences differ in exactly 1 element
		if diff_counter == 1:
			idx: int = diff.pop()
			base_token: Token = conllu_all[indices[i]].tokens[idx]
			ref_token: Token = conllu_all[indices[j]].tokens[idx]
			pos_tags: list[str] = [base_token[pos_key], ref_token[pos_key]]
			# exclude the pair if not at least one the replaced words is a noun
			if "NOUN" not in pos_tags:
				return sim_dict
			base_lemma: str = base_token[lemma_key]
			# the difference is lexical, not just morphological; syntactic function is different
			if base_lemma != ref_token[lemma_key] and base_token[deprel_key] == ref_token[deprel_key]:
				base_sent: str = tok_lists_concat[i]
				target_sentence: str = tok_lists_concat[j]
				sim_dict[base_sent] = sim_dict.get(base_sent, []) + [target_sentence]
				print(f"{base_sent}\n{target_sentence}")
		return sim_dict

	conllu_all: List[TokenList] = pickle.load(open(proiel_pickle, "rb"))  # [:4000]
	tok_lists: list[np.array] = []
	sent_ids: list[str] = []
	indices: list[int] = []
	lemma_key: str = "lemma"
	pos_key: str = "upos"
	deprel_key: str = "deprel"
	for i in range(len(conllu_all)):
		# exclude short sentences
		if len(conllu_all[i].tokens) < 4:
			continue
		new_toks: np.array = np.array([x["form"] for x in conllu_all[i].tokens])
		tok_lists.append(new_toks)
		sent_ids.append(conllu_all[i].metadata["sent_id"])
		indices.append(i)
	sim_dict: dict[str, list[str]] = dict()
	sent_lengths: list[int] = [len(x) for x in tok_lists]
	tok_lists_concat: list[str] = [" ".join(tok_lists[i]) + f" ({sent_ids[i]})" for i in range(len(tok_lists))]
	for i in tqdm(range(len(tok_lists))):
		for j in range(len(tok_lists)):
			new_sims: dict[str, list[str]] = process_sentence()
			if new_sims:
				for key in new_sims:
					sim_dict[key] = sim_dict.get(key, []) + new_sims[key]
	with open(similar_syntax_of_nouns_path, "w+") as f:
		json.dump(sim_dict, f)


def get_sentence_pair_id(id_1: str, id_2: str) -> str:
	ids: list[str] = [id_1, id_2]
	ids.sort()
	return "".join(ids)


raw_dataset_path: str = os.path.join(Config.data_dir, "corpus_corporum", "corpus_corporum_tokenized.tsv")
ud_path: str = os.path.join(Config.data_dir, "universal_dependencies")
french_path: str = os.path.join(ud_path, "fr_ftb-ud-train.conllu")
german_path: str = os.path.join(ud_path, "de_hdt-ud-dev.conllu")
database_url: str = 'postgresql://postgres:postgres@localhost:5432/postgres'
similar_syntax_of_nouns_path: str = os.path.abspath("similar_syntax_of_nouns.json")
if __name__ == '__main__':
	create_proiel_cache(Config.proiel_dir, Config.proiel_pickle)
	# create_dataset_proiel(proiel_pickle)
	# create_dataset_pars()
	# calculate_verb_to_noun_ratio_proiel(proiel_pickle)
	# calculate_verb_to_noun_ratio_ud(french_path, german_path)
	calculate_contexts_raw(Config.proiel_pickle)
# calculate_contexts_dvs(proiel_pickle)
# calculate_types_per_lemma(proiel_pickle, french_path, german_path)
# find_lexically_identical_sentences(proiel_pickle)
# find_syntactically_identical_sentences(proiel_pickle, database_url)
# find_syntactically_identical_sentences_annis()
# find_syntactically_identical_sentences_with_nouns(proiel_pickle, similar_syntax_of_nouns_path)

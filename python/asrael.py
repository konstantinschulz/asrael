import codecs
import json
import math
import os
import pickle
from collections import Counter
from multiprocessing import Manager, cpu_count
from multiprocessing.pool import Pool
from queue import Queue
from typing import List, Dict, Set, Tuple
import csv
from zipfile import ZipFile
import matplotlib.pyplot as plt

import requests
import conllu as conllu
import nltk
from cltk.tokenize.word import WordTokenizer
from conllu import TokenList
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
import torch
from gru import ConcatPoolingGRUAdaptive


class WordNetPaths:
	def __init__(self, relations_path: str, lemmas_synsets_path: str, lemmas_path: str, synsets_path: str):
		self.relations_path: str = relations_path
		self.lemmas_synsets_path: str = lemmas_synsets_path
		self.lemmas_path: str = lemmas_path
		self.synsets_path: str = synsets_path


class Lemma:
	def __init__(self, values: List[str]):
		self.lemma: str = values[0]
		self.morphology: str = values[1]
		self.uri: str = values[2]
		self.principal_parts: str = values[3]
		self.irregular_forms: str = values[4]
		self.alternative_forms: str = values[5]
		self.pronunciation: str = values[6]
		self.synsets: Set[str] = set()


class LemmaSynsetMap:
	def __init__(self, values: List[str]):
		self.uri: str = values[0]
		self.literal: str = values[1]
		self.metonymic: str = values[2]
		self.metaphoric: str = values[3]


class Relation:
	def __init__(self, values: List[str]):
		self.type: str = values[0]
		self.source: str = values[1]
		self.target: str = values[2]
		self.w_source: str = values[3]
		self.w_target: str = values[4]


class WordNet:
	def __init__(self, directory_path: str, wnp: WordNetPaths):
		lemmata: List[List[str]] = \
			[x for x in csv.reader(open(os.path.join(directory_path, wnp.lemmas_path), newline=''))][1:]
		self.lemmata: List[Lemma] = [Lemma(x) for x in lemmata]
		self.lemma_uri_to_idx: Dict[str, int] = {self.lemmata[i].uri: i for i in range(len(self.lemmata))}
		lemmata_synsets: List[List[str]] = \
			[x for x in csv.reader(open(os.path.join(directory_path, wnp.lemmas_synsets_path), newline=''))][1:]
		self.lemma_synset_maps: List[LemmaSynsetMap] = [LemmaSynsetMap(x) for x in lemmata_synsets]
		for lsm in self.lemma_synset_maps:
			lemma: Lemma = self.get_lemma_by_uri(lsm.uri)
			lemma.synsets = set(lsm.literal.split())
		self.synset_to_lemma_map: Dict[str, Set[str]] = {}
		for lemma in self.lemmata:
			for synset in lemma.synsets:
				if synset not in self.synset_to_lemma_map:
					self.synset_to_lemma_map[synset] = set()
				self.synset_to_lemma_map[synset].add(lemma.uri)
		relations: List[List[str]] = \
			[x for x in csv.reader(open(os.path.join(directory_path, wnp.relations_path), newline=''))][1:]
		self.relations: List[Relation] = [Relation(x) for x in relations]

	def get_hypernym_relations(self) -> List[Relation]:
		return [x for x in self.relations if x.type == "@"]

	def get_hyponym_relations(self) -> List[Relation]:
		return [x for x in self.relations if x.type == "~"]

	def get_hyponym_to_lemmata_map(self):
		hyponym_to_hypernym_map: Dict[str, Set[str]] = self.get_hyponym_to_synsets_map()
		lemma_to_hypernym: Dict[str, Set[str]] = {}
		for lemma in tqdm(self.lemmata):
			if lemma.uri not in lemma_to_hypernym:
				lemma_to_hypernym[lemma.lemma] = set()
			for synset_id in lemma.synsets:
				if synset_id not in hyponym_to_hypernym_map:
					continue
				for hypernym_synset_id in hyponym_to_hypernym_map[synset_id]:
					if hypernym_synset_id not in self.synset_to_lemma_map:
						continue
					for hypernym_lemma_id in self.synset_to_lemma_map[hypernym_synset_id]:
						hypernym_lemma: Lemma = self.get_lemma_by_uri(hypernym_lemma_id)
						lemma_to_hypernym[lemma.lemma].add(hypernym_lemma.lemma)
		return lemma_to_hypernym

	def get_hyponym_to_synsets_map(self) -> Dict[str, Set[str]]:
		hyponym_to_hypernym: Dict[str, Set[str]] = {}
		for hyponymy_relation in self.get_hyponym_relations():
			if hyponymy_relation.target not in hyponym_to_hypernym:
				hyponym_to_hypernym[hyponymy_relation.target] = set()
			# target is a hyponym of source
			hyponym_to_hypernym[hyponymy_relation.target].add(hyponymy_relation.source)
		return hyponym_to_hypernym

	def get_lemma_by_uri(self, lemma_uri: str) -> Lemma:
		return self.lemmata[self.lemma_uri_to_idx[lemma_uri]]


def add_hypernyms(work_queue: Queue, lemma_to_hypernym: Dict[str, Set[str]] = None, q: Queue = None,
                  finished_queue: Queue = None) -> None:
	while True:
		sentence: List[str]
		line_number: int
		(sentence, line_number) = work_queue.get()
		if line_number < 0:
			break
		hyponym_set: Set[Tuple[str, str]] = set()
		for i in range(len(sentence)):
			if sentence[i] not in lemma_to_hypernym:
				continue
			lemmata_sub_set: Set[str] = set(sentence)
			lemmata_sub_set.remove(sentence[i])
			overlap: Set[str] = lemmata_sub_set.intersection(lemma_to_hypernym[sentence[i]])
			while len(overlap) > 0:
				target_lemma: str = overlap.pop()
				target_lemma_idx: int = sentence.index(target_lemma)
				# base lemma is hyponym of target lemma
				hyponym_set.add((sentence[i], sentence[target_lemma_idx]))
		if not len(hyponym_set):
			finished_queue.put(1)
			continue
		content: str = f"{line_number}\t" + (" ".join([",".join(x) for x in hyponym_set]) + "\n")
		q.put(content)
		finished_queue.put(1)


def apply_model(max_epochs: int, models_dir: str, model: ConcatPoolingGRUAdaptive):
	checkpointer: ModelCheckpoint = ModelCheckpoint(models_dir, 'cc_detect', save_interval=1, n_saved=10,
	                                                create_dir=True,
	                                                save_as_state_dict=True, require_empty=False)
	model.trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'hyponyms': model})
	print("Running model...")
	model.trainer.run(model.train_dl, max_epochs=max_epochs)
	torch.cuda.empty_cache()


def build_corpus(raw_text_path: str, processed_text_path: str) -> None:
	if not os.path.exists(raw_text_path):
		print("Downloading corpus...")
		zip_file_path: str = raw_text_path + ".zip"
		response: requests.Response = requests.get("https://box.hu-berlin.de/f/056b874a12cf44de82ab/?dl=1", stream=True)
		total_length: int = int(response.headers.get("content-length"))
		done_count: int = 0
		chunk_size: int = 1024
		with open(zip_file_path, "wb+") as f:
			for data in tqdm(response.iter_content(chunk_size=chunk_size), total=math.ceil(total_length // chunk_size),
			                 unit="MB", unit_scale=0.001):  # math.ceil(total_length // chunk_size)
				done_count += len(data)
				f.write(data)
		print("Extracting corpus...")
		zip_file: ZipFile = ZipFile(zip_file_path)
		file_path_parts: Tuple[str, str] = os.path.split(raw_text_path)
		zip_file.extract(file_path_parts[1], file_path_parts[0])
		zip_file.close()
	print("Segmenting and tokenizing corpus...")
	raw_text: str
	with open(raw_text_path) as f:
		raw_text = f.read()
	language: str = "latin"
	raw_sentences: List[str] = nltk.sent_tokenize(raw_text, language=language)
	del raw_text
	word_tokenizer = WordTokenizer(language)
	with open(processed_text_path, "a+") as f:
		raw_text_tokenized = []
		for sent in tqdm(raw_sentences):
			raw_text_tokenized.append(word_tokenizer.tokenize(sent))
			if len(raw_text_tokenized) == 1000:
				for sentence in raw_text_tokenized:
					f.write("\t".join(sentence) + "\n")
				raw_text_tokenized = []


def build_dataset_file(hyponyms_detected_path: str, dataset_path: str, processed_text_path: str):
	sent_to_detect_dict: Dict[int, int] = {}
	with open(hyponyms_detected_path) as f:
		while True:
			line: str = f.readline()
			if not line:
				break
			sent_to_detect_dict[int(line.split("\t")[0])] = 1
	max_sent_id: int = max(sent_to_detect_dict.keys())
	chunk_size: int = max_sent_id // 100
	with open(dataset_path, "a+") as f:
		with open(processed_text_path) as f2:
			sent_id: int = 0
			while True:
				sent_id += 1
				line: str = f2.readline().strip()
				if not line:
					break
				if sent_id % chunk_size == 0:
					print(f"{sent_id * 100 // max_sent_id}%")
				detect_value: int = sent_to_detect_dict.get(sent_id, 0)
				sentence: str = " ".join(line.split("\t"))
				f.write(f"{sentence}\t{detect_value}\n")


def detect_hypernyms(raw_text_path: str, cache_dir: str, wn: WordNet, hyponyms_detected_path: str,
                     processed_text_path: str):
	lemma_to_hypernym: Dict[str, Set[str]] = map_lemmata_to_hypernyms_bidirectional(wn)
	with Manager() as manager:
		with Pool() as pool:  # processes=7
			d = manager.dict(lemma_to_hypernym)
			work_queue: Queue = manager.Queue()
			result_queue: Queue = manager.Queue()
			finished_queue: Queue = manager.Queue()
			pool.apply_async(file_writer, args=(hyponyms_detected_path, result_queue))
			for i in range(cpu_count() + 1):
				pool.apply_async(add_hypernyms, args=(work_queue, d, result_queue, finished_queue))
			line_number: int = 0
			for sent in generate_tokenized_segments(raw_text_path, cache_dir, processed_text_path):
				# if line_number == 500:
				#     break
				if line_number % 100000 == 0:
					print(f"{line_number} in")
				line_number += 1
				work_queue.put((sent, line_number))
			for i in range(cpu_count() + 1):
				# now we are done, kill the consumers
				work_queue.put(([], -1))
			finished_count: int = 0
			while finished_count < line_number:
				if finished_count % 100000 == 0:
					print(f"{finished_count} out")
				finished_count += finished_queue.get()
			# now we are done, kill the file writer
			result_queue.put("XXX")
			pool.close()
			pool.join()


def detect_hypernyms_proiel(cache_dir: str, conll_dir: str, lemma_to_hypernym: Dict[str, Set[str]]):
	conll: List[TokenList] = []
	for x in tqdm([x for x in os.listdir(conll_dir) if x.endswith(".conllu")]):
		conll += conllu.parse(open(os.path.join(conll_dir, x)).read())
	train_data: List[List[str]] = []
	for sent in tqdm(conll):
		lemmata: List[str] = [tok["lemma"] for tok in sent]
		for i in range(len(sent)):
			if lemmata[i] not in lemma_to_hypernym:
				continue
			lemmata_sub_set: Set[str] = set(lemmata)
			lemmata_sub_set.remove(lemmata[i])
			overlap: Set[str] = lemmata_sub_set.intersection(lemma_to_hypernym[lemmata[i]])
			if len(overlap) == 0:
				continue
			target_lemma: str = overlap.pop()
			target_lemma_idx: int = lemmata.index(target_lemma)
			# base lemma is hyponym of target lemma
			train_data.append(
				[" ".join([tok["form"] for tok in sent]), sent[i]["form"], sent[target_lemma_idx]["form"]])
	json.dump(train_data, open(os.path.join(cache_dir, "proiel_hyponyms.json"), "w+"))


def evaluate_model(model: ConcatPoolingGRUAdaptive, models_dir: str):
	def make_predictions(model: ConcatPoolingGRUAdaptive) -> List[Tuple[int, int]]:
		predictions: List[Tuple[int, int]] = []
		with torch.no_grad():
			while True:
				batch: Tuple[torch.Tensor, torch.Tensor, List[int]] = next(x for x in model.val_dl)
				x, y, lengths = batch
				x, y = x.to(model.device), y.to(model.device)
				y_pred: torch.Tensor = model(x, lengths)
				new_predictions: List[int] = [int(x) for x in torch.max(y_pred, dim=1)[1]]
				for i in range(len(new_predictions)):
					# skip true negatives
					if new_predictions[i] + int(y[i]) > 0:
						predictions.append((new_predictions[i], int(y[i])))
					if len(predictions) == 1000:
						return predictions

	model_path: str = os.path.join(models_dir, "cc_detect_hyponyms_3.pth")
	model.load_state_dict(torch.load(model_path))
	model.eval()
	predictions: List[Tuple[int, int]] = make_predictions(model)
	precision_counter: int = 0
	recall_counter: int = 0
	correct: int = 0
	for pred in predictions:
		# true positives
		if pred[0] == pred[1] == 1:
			correct += 1
			precision_counter += 1
			recall_counter += 1
		# false negatives
		elif pred[0] > pred[1]:
			precision_counter += 1
		# false positives
		else:
			recall_counter += 1
	precision: float = (correct / precision_counter)
	print(f"Precision: {precision}")
	recall: float = correct / recall_counter
	print(f"Recall: {recall}")
	print(f"F1: {2 * (precision * recall) / (precision + recall)}")


def evaluate_heuristic_model(hyponyms_detected_path: str, plots_dir: str):
	counter: Counter = Counter()
	for hyponym_pair in get_hyponym_pairs(hyponyms_detected_path):
		counter.update(hyponym for hyponym in hyponym_pair.split(","))
	labels: List[str]
	values: List[int]
	labels, values = zip(*counter.most_common())
	occurrence_sum: int = sum(values)
	occurrence_count: int = 0
	percentages: List[float] = []
	for value in values:
		occurrence_count += value
		percentages.append(occurrence_count / occurrence_sum * 100)
	fig: Figure
	ax: Axes
	fig, ax = plt.subplots()
	ax.scatter(list(range(len(percentages))), percentages)
	ax.set_title('Frequency distribution of hyponym lemmata in Corpus Corporum (plain)', size=12)
	plt.xlabel('No. of lemmata')
	plt.ylabel('Coverage (%)')
	# plt.show()
	fig.savefig(os.path.join(plots_dir, "freq_dist_hyponyms_cc_plain.png"), dpi=600)


def file_writer(file_path: str, q: Queue) -> str:
	with open(file_path, 'a+') as f:
		next_chunks: List[str] = []
		while True:
			content: str = q.get()
			if content:
				if content == "XXX":
					f.write("".join(next_chunks))
					f.flush()
					break
				next_chunks.append(content)
				if len(next_chunks) > 1000:
					f.write("".join(next_chunks))
					f.flush()
					next_chunks = []
	return ""


def generate_tokenized_segments(raw_text_path: str, cache_dir: str, processed_text_path: str) -> List[List[str]]:
	raw_text_tokenized: List[List[str]]
	if os.path.exists(processed_text_path):
		with open(processed_text_path) as f:
			while True:
				line: str = f.readline()
				if not line:
					break
				yield line.split("\t")[:-1]
	else:
		build_corpus(raw_text_path, processed_text_path)
		for sent in generate_tokenized_segments(raw_text_path, cache_dir, processed_text_path):
			yield sent
	return []


def get_hypernyms(target_synset: str, relations: List[List[str]], hyponym_dict: Dict[str, Set[int]],
                  synsets_list: List[List[str]], synsets_list_dict: Dict[str, int], history: List[str]) -> None:
	# prevent circularity
	if target_synset not in hyponym_dict or synsets_list[synsets_list_dict[target_synset[2:]]][2] in history:
		return
	history.append(synsets_list[synsets_list_dict[target_synset[2:]]][2])
	for index in hyponym_dict[target_synset]:
		print(synsets_list[synsets_list_dict[target_synset[2:]]][2], "-->",
		      synsets_list[synsets_list_dict[relations[index][1][2:]]][2])
		get_hypernyms(relations[index][1], relations, hyponym_dict, synsets_list, synsets_list_dict, history)


def get_hyponym_pairs(hyponyms_detected_path: str) -> List[str]:
	hyponym_pairs: List[str] = []
	with open(hyponyms_detected_path) as f:
		for line in tqdm(f.read().split("\n")):
			if not line:
				break
			hyponym_pairs += line.split("\t")[1].split()
	return hyponym_pairs


def get_synsets(lemma: str, wordnet_path: str, wnp: WordNetPaths) -> List[str]:
	lemmata: List[List[str]] = [x for x in csv.reader(open(os.path.join(wordnet_path, wnp.lemmas_path), newline=''))][
	                           1:]
	lemmata_dict: Dict[str, int] = {v[0]: i for i, v in enumerate(lemmata)}
	if lemma not in lemmata_dict:
		return []
	lemma_id: str = lemmata[lemmata_dict[lemma]][2]
	lemmata_synsets_path: str = os.path.join(wordnet_path, wnp.lemmas_synsets_path)
	lemmata_synsets: List[List[str]] = [x for x in csv.reader(open(lemmata_synsets_path, newline=''))][1:]
	lemmata_synsets_dict: Dict[str, int] = {v[0]: i for i, v in enumerate(lemmata_synsets)}
	return lemmata_synsets[lemmata_synsets_dict[lemma_id]][1].split()


def get_token_count(corpus_path: str, is_proiel: bool):
	tok_count: int = 0
	if is_proiel:
		for file in tqdm([x for x in os.listdir(corpus_path) if x.endswith(".conllu")]):
			with open(os.path.join(corpus_path, file)) as f:
				sentences: List[TokenList] = conllu.parse(f.read())
				tok_count += sum(len(x.tokens) for x in sentences)
		return tok_count
	with open(corpus_path) as f:
		line: str = f.readline()
		while line != "":
			tok_count += len(line.split())
			line = f.readline()
	return tok_count


def get_type_count(hyponyms_detected_path: str) -> int:
	types: Set[str] = set()
	for hyponym_pair in get_hyponym_pairs(hyponyms_detected_path):
		for hyponym in hyponym_pair.split(","):
			types.add(hyponym)
	return len(types)


def map_lemmata_to_hypernyms_bidirectional(wn: WordNet) -> Dict[str, Set[str]]:
	synset_to_lemma: Dict[str, Set[str]] = {}
	for lemma in wn.lemmata:
		for synset_id in lemma.synsets:
			if synset_id not in synset_to_lemma:
				synset_to_lemma[synset_id] = set()
			synset_to_lemma[synset_id].add(lemma.uri)
	# TODO: SOME LEMMATA APPEAR IN BOTH COMBINATIONS (I.E. AS HYPONYM > HYPERNYM AND HYPERNYM > HYPONYM). WHICH CASE IS TRUE?
	hyponym_to_hypernym_map: Dict[str, Set[str]] = wn.get_hyponym_to_synsets_map()
	for hypernymy_relation in wn.get_hypernym_relations():
		if hypernymy_relation.source not in hyponym_to_hypernym_map:
			# source is a hyponym of target
			hyponym_to_hypernym_map[hypernymy_relation.source] = set()
		hyponym_to_hypernym_map[hypernymy_relation.source].add(hypernymy_relation.target)
	lemma_to_hypernym: Dict[str, Set[str]] = {}
	for lemma in tqdm(wn.lemmata):
		if lemma.uri not in lemma_to_hypernym:
			lemma_to_hypernym[lemma.uri] = set()
		for synset_id in lemma.synsets:
			if synset_id not in hyponym_to_hypernym_map:
				continue
			for hypernym_synset_id in hyponym_to_hypernym_map[synset_id]:
				if hypernym_synset_id not in synset_to_lemma:
					continue
				for hypernym_lemma_id in synset_to_lemma[hypernym_synset_id]:
					lemma_to_hypernym[lemma.uri].add(hypernym_lemma_id)
	return lemma_to_hypernym


def print_hypernyms(wordnet_path: str, wnp: WordNetPaths, key_word: str):
	relations: List[List[str]] = [x for x in
	                              csv.reader(open(os.path.join(wordnet_path, wnp.relations_path), newline=''))][1:]
	hyponym_dict: Dict[str, Set[int]] = {}
	for i, v in enumerate(x for x in relations if x[0] == "~"):
		if v[2] not in hyponym_dict:
			hyponym_dict[v[2]] = set()
		hyponym_dict[v[2]].add(i)
	synset_ids: List[str] = get_synsets(key_word, wordnet_path, wnp)
	synsets_list: List[List[str]] = [x for x in
	                                 csv.reader(open(os.path.join(wordnet_path, wnp.synsets_path), newline=''))][1:]
	synsets_list_dict: Dict[str, int] = {v[1]: i for i, v in enumerate(synsets_list)}
	for synset_id in synset_ids:
		print("\nHypernyms for", synsets_list[synsets_list_dict[synset_id[2:]]][2], "\n")
		get_hypernyms(synset_id, relations, hyponym_dict, synsets_list, synsets_list_dict, [])


def search_proiel(sentence_id: str):
	proiel_dir: str = "../data/proiel_conllu"
	for file in tqdm([x for x in os.listdir(proiel_dir) if x.endswith(".conllu")]):
		with open(os.path.join(proiel_dir, file)) as f:
			sentences: List[TokenList] = conllu.parse(f.read())
			for sentence in sentences:
				if sentence.metadata["sent_id"] == sentence_id:
					print(sentence)
					return 


def train_latin_tokenizer(raw_text_path: str) -> None:
	# Make a new Tokenizer
	tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
	# Read in training corpus (one example: Slovene)
	# restrict corpus size to prevent memory overflow
	text = codecs.open(raw_text_path, "Ur", "iso-8859-2").read()[:(100 * 1000 * 1000)]
	# Train tokenizer
	tokenizer.train(text, verbose=True)
	# Dump pickled tokenizer
	with open("latin.pickle", "wb+") as f:
		pickle.dump(tokenizer, f)


data_dir: str = "../data"
wordnet_path: str = os.path.join(data_dir, "latinwordnet2-master", "csv")
conll_dir: str = os.path.join(data_dir, "proiel_conllu")
cache_dir: str = os.path.join(data_dir, "corpus_corporum")
raw_text_path: str = os.path.join(cache_dir, "corpus_corporum.txt")
processed_text_path: str = os.path.join(cache_dir, "corpus_corporum_tokenized.tsv")
hyponyms_detected_path: str = os.path.join(cache_dir, "corpus_corporum_hyponyms.tsv")
dataset_path: str = os.path.join(cache_dir, "dataset_detect.tsv")
wnp: WordNetPaths = WordNetPaths("lwn2-n-relations.csv", "lwn2-n-lemmas-synsets.csv", "lwn2-n-lemmas.csv",
                                 "lwn2-n-synsets.csv")
models_dir: str = "models"
plots_dir: str = os.path.join(data_dir, "plots")
batch_size: int = 100  # 10000
dataset_max_size: int = 10000
validation_split: float = .05
shuffle_dataset: bool = True
random_seed: int = 42
embedding_dim: int = 100  # 100
n_hidden: int = 256  # 256
n_out: int = 2
max_epochs: int = 3
dropout: float = 0.0
device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# wn: WordNet = WordNet(wordnet_path, wnp)
# nltk.download('punkt')
# train_latin_tokenizer(raw_text_path)
# detect_hypernyms(raw_text_path, cache_dir, wn, hyponyms_detected_path, processed_text_path)
# build_dataset_file(hyponyms_detected_path, dataset_path, processed_text_path)
# lemma_to_hypernym: Dict[str, Set[str]] = map_lemmata_to_hypernyms_bidirectional(wn)
# lemma_to_hypernym: Dict[str, Set[str]] = wn.get_hyponym_to_lemmata_map()
# json.dump({k: list(v) for k, v in lemma_to_hypernym.items()}, open("lemma_to_hypernym_unidirectional.json", "w+"))
# detect_hypernyms_proiel("../data", conll_dir, lemma_to_hypernym)
# print_hypernyms(wordnet_path, wnp, "gladius")
# search_proiel("53467")
# print(get_token_count(processed_text_path, False))
# print(get_token_count(conll_dir, True))
# model: ConcatPoolingGRUAdaptive = \
#     ConcatPoolingGRUAdaptive(embedding_dim, n_hidden, n_out, device, dropout, dataset_path, dataset_max_size,
#                              batch_size, validation_split, shuffle_dataset, random_seed).to(device)
# apply_model(max_epochs, models_dir, model)
# evaluate_model(model, models_dir)
# print(get_type_count(hyponyms_detected_path))
# evaluate_heuristic_model(hyponyms_detected_path, plots_dir)
lemma_to_hypernym_raw: Dict[str, List[str]] = json.load(open("lemma_to_hypernym_unidirectional.json"))
lemma_to_hypernym: Dict[str, Set[str]] = {}
for key in lemma_to_hypernym_raw:
	lemma_to_hypernym[key] = set(lemma_to_hypernym_raw[key])
line_number: int = 0
hyponym_set: List[Tuple[int, str, str]] = []
for sentence in generate_tokenized_segments(raw_text_path, cache_dir, processed_text_path):
	if line_number == 500:
		break
	if line_number % 100000 == 0:
		print(f"{line_number} in")
	line_number += 1
	sentence: List[str]
	line_number: int
	if line_number < 0:
		break
	lemmata: Set[str] = set(sentence)
	for i in range(len(sentence)):
		if sentence[i] not in lemma_to_hypernym:
			continue
		lemmata.remove(sentence[i])
		overlap: Set[str] = lemmata.intersection(lemma_to_hypernym[sentence[i]])
		while len(overlap) > 0:
			target_lemma: str = overlap.pop()
			# base lemma is hyponym of target lemma
			hyponym_set.append((line_number, sentence[i], target_lemma))
		lemmata.add(sentence[i])
a = 0

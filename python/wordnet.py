from latinwordnet import LatinWordNet
from latinwordnet.latinwordnet import Lemmas, Synsets


class Synset:
	def __init__(self, synset_dict: dict):
		self.gloss: str = synset_dict["gloss"]
		self.offset: str = synset_dict["offset"]
		self.pos: str = synset_dict["pos"]


lwn: LatinWordNet = LatinWordNet()
lemmas: Lemmas = lwn.lemmas(lemma="suprascriptio", pos="n")  # uerbum sermo scriba publicanus ethnicus
synset_results: list[dict] = lemmas.synsets
target_synset: Synset = Synset(synset_results[0]["synsets"]["literal"][0])
print(f"Synset: {target_synset.gloss}")
print(f"{[x['lemma'] for x in lwn.synsets(**target_synset.__dict__).lemmas[0]['lemmas']['literal']]}")
relations: dict[str, list] = lwn.synsets(**target_synset.__dict__).relations
hypernym_synsets: list[dict] = relations["@"]


def process_hypernyms(hypernym_synsets: list[dict]):
	""" Cycles through all hypernym synsets and lists the corresponding lemmata for each one. """
	for hs in hypernym_synsets:
		synset: Synsets = lwn.synsets(**Synset(hs).__dict__)
		print(f"Synset: {synset.gloss}")
		lemma: dict = synset.lemmas[0]["lemmas"]
		lemmata: list[str] = [x['lemma'] for x in lemma['literal']]
		print(f"{lemmata}")
		new_hypernym_synsets: list[dict] = synset.relations["@"]
		if new_hypernym_synsets:
			process_hypernyms(new_hypernym_synsets)


process_hypernyms(hypernym_synsets)
a = 0

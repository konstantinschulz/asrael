from latinwordnet import LatinWordNet
from latinwordnet.latinwordnet import Lemmas, Synsets


class LemmaWithSynsetIndices:
    def __init__(self, lemma: str, synset_idx: int = 0, literal_idx: int = 0):
        self.lemma: str = lemma
        self.synset_idx: int = synset_idx
        self.literal_idx: int = literal_idx


class SynsetHelper:
    def __init__(self, synset_dict: dict):
        self.gloss: str = synset_dict["gloss"]
        self.offset: str = synset_dict["offset"]
        self.pos: str = synset_dict["pos"]


lwn: LatinWordNet = LatinWordNet()
lemmata_with_indices: list[tuple[LemmaWithSynsetIndices, LemmaWithSynsetIndices]] = [
    (LemmaWithSynsetIndices("lectus", 1, 2), LemmaWithSynsetIndices("crabattus", literal_idx=3)),  # [3, 6, 8]
    (LemmaWithSynsetIndices("caelum", literal_idx=12), LemmaWithSynsetIndices("deus", literal_idx=2)),
    (LemmaWithSynsetIndices("puer", literal_idx=2), LemmaWithSynsetIndices("discipulus", literal_idx=1)),
    (LemmaWithSynsetIndices("sermo", literal_idx=2), LemmaWithSynsetIndices("uerbum", literal_idx=1)),
    (LemmaWithSynsetIndices("suprascriptio", literal_idx=16), LemmaWithSynsetIndices("inscriptio", literal_idx=0)),
    (LemmaWithSynsetIndices("feria", literal_idx=0), LemmaWithSynsetIndices("dies", literal_idx=0)),
    (LemmaWithSynsetIndices("publicanus", literal_idx=0), LemmaWithSynsetIndices("ethnicus", literal_idx=1)),
    (LemmaWithSynsetIndices("tabernaculum", literal_idx=0), LemmaWithSynsetIndices("nidus", literal_idx=0)),
    # no proper entries for Iesus, nubes, is/ea/id
]


def find_common_hypernyms(lemmata_with_indices: list[tuple[LemmaWithSynsetIndices, LemmaWithSynsetIndices]]):
    for lemma_pair in lemmata_with_indices:
        hypernyms_1: dict[int, str] = get_hypernyms_for_lemma(lemma_pair[0])
        hypernyms_2: dict[int, str] = get_hypernyms_for_lemma(lemma_pair[1])
        indices: tuple[int, int]
        ref_idx: int = -1
        for key in hypernyms_1:
            ref_idx: int = next((x for x in hypernyms_2 if hypernyms_2[x] == hypernyms_1[key]), -1)
            if ref_idx >= 0:
                lemma_1: str = lemma_pair[0].lemma
                lemma_2: str = lemma_pair[1].lemma
                print(f"{lemma_1} vs. {lemma_2}: {key}/{len(hypernyms_1)} == {ref_idx}/{len(hypernyms_2)}")
                break
        if ref_idx < 0:
            print("No common hypernym")
        # print(hypernyms_1)
        # print(hypernyms_2)


def get_hypernyms_for_lemma(lwsi: LemmaWithSynsetIndices) -> dict[int, str]:
    """ Retrieves all relevant hypernyms for a given lemma across the entire hierarchy. """
    lemmas: Lemmas = lwn.lemmas(lemma=lwsi.lemma, pos="n")  # uerbum sermo scriba publicanus ethnicus
    synset_results: list[dict] = lemmas.synsets
    target_synset: SynsetHelper = SynsetHelper(synset_results[lwsi.synset_idx]["synsets"]["literal"][lwsi.literal_idx])
    # print(f"Synset: {target_synset.gloss}")
    # print(f"{[x['lemma'] for x in lwn.synsets(**target_synset.__dict__).lemmas[0]['lemmas']['literal']]}")
    hypernym_synsets: list[Synsets] = get_hypernym_synsets(lwn.synsets(**target_synset.__dict__))
    hypernyms: dict[int, str] = {0: target_synset.gloss}
    hypernyms.update(process_hypernyms(hypernym_synsets))
    return hypernyms


def get_hypernym_synsets(base_synset: Synsets) -> list[Synsets]:
    """ Retrieves all direct hypernym synsets for a given base synset."""
    # ignore entries in Italian etc.
    synset_dicts: list[dict] = [x for x in base_synset.relations["@"] if x["language"]["abbrev"] == "Eng"]
    return [lwn.synsets(**SynsetHelper(synset_dict).__dict__) for synset_dict in synset_dicts]


def process_hypernyms(hypernym_synsets: list[Synsets]) -> dict[int, str]:
    """ Cycles through all hypernym synsets and lists the corresponding lemmata for each one. """
    if len(hypernym_synsets) > 1:
        print(f"{len(hypernym_synsets)} more synset(s) to process: {hypernym_synsets[0].gloss}")
    hypernyms: dict[int, str] = dict()
    for hs in hypernym_synsets:
        # print(f"Synset: {synset.gloss}")
        hypernyms[len(hypernyms) + 1] = hs.gloss
        # lemma: dict = hs.lemmas[0]["lemmas"]
        # lemmata: list[str] = [x['lemma'] for x in lemma['literal']]
        # print(f"{lemmata}")
        new_hypernym_synsets: list[Synsets] = get_hypernym_synsets(hs)
        if new_hypernym_synsets:
            new_hypernyms: dict[int, str] = process_hypernyms(new_hypernym_synsets)
            for key in new_hypernyms:
                hypernyms[len(hypernyms) + 1] = new_hypernyms[key]
    return hypernyms


# process_hypernyms(hypernym_synsets)
# get_hypernyms_for_lemma("suprascriptio")
find_common_hypernyms(lemmata_with_indices)
a = 0

import glob
import os
import random
import statistics
from collections import Counter, defaultdict

data_dir = os.path.join(os.path.abspath(".."), "data")
sents_lemmatized: list[list[str]] = []
# extract lemma counts from the all CONLL-U files in the given directory
for f in sorted(glob.glob(os.path.join(data_dir, "*.conllu"))):
    current = []
    for line in open(f, encoding="utf8"):
        line = line.rstrip("\n")
        if not line:
            if current:
                sents_lemmatized.append(current)
                current = []
            continue
        if line.startswith("#"):
            continue
        token_parts = line.split("\t")
        if "-" in token_parts[0] or "." in token_parts[0]:
            continue
        if token_parts[3] == "PUNCT":
            continue
        current.append(token_parts[2])
    if current:
        sents_lemmatized.append(current)
sentence_lengths = [len(s) for s in sents_lemmatized]
print(f"Standard deviation of sentence lengths: {statistics.stdev(sentence_lengths):.2f}")
token_count = sum(sentence_lengths)
sentence_count = len(sents_lemmatized)
lemma_counter = Counter(lemma for sent in sents_lemmatized for lemma in sent)
# dis legomena are those that occur exactly twice in the corpus
dis: set[str] = {lemma for lemma, count in lemma_counter.items() if count == 2}


def count_dis_legomena_in_same_sentence(sents):
    loc: defaultdict[str, set[int]] = defaultdict(set)
    for idx, sent in enumerate(sents):
        for lemma in sent:
            if lemma in dis:
                loc[lemma].add(idx)
    return sum(1 for l in dis if len(loc[l]) == 1)


observed = count_dis_legomena_in_same_sentence(sents_lemmatized)
p0 = sum(n * (n - 1) for n in sentence_lengths) / (token_count * (token_count - 1))
print(
    f"sentences={sentence_count} tokens={token_count} lemmata={len(lemma_counter)} dis={len(dis)} mean_len={token_count / sentence_count:.2f}"
)
print(f"observed same-sentence dis legomena={observed} share={observed / len(dis):.4f}")
print(
    f"null probability per pair={p0:.3e} naive 1/S={1 / sentence_count:.3e} expected count={len(dis) * p0:.3f} ratio={observed / len(dis) / p0:.0f}"
)
tokens = [l for s in sents_lemmatized for l in s]
random.seed(0)
result = []
for _ in range(1000):
    random.shuffle(tokens)
    out = []
    i = 0
    for n in sentence_lengths:
        out.append(tokens[i : i + n])
        i += n
    result.append(count_dis_legomena_in_same_sentence(out))
print(
    f"permutation: mean={sum(result) / len(result):.3f} max={max(result)} >=observed: {sum(r >= observed for r in result)}/1000"
)

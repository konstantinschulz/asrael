import json
import os.path
import pickle
from collections import Counter
from statistics import linear_regression, LinearRegression
from typing import Union

import numpy as np
import torch
from conllu import TokenList
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, IntervalStrategy, \
    BatchEncoding, DataCollatorForLanguageModeling, pipeline, Pipeline, \
    GPT2TokenizerFast, OPTForCausalLM, OPTConfig
from transformers.integrations import TensorBoardCallback
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import PaddingStrategy

from python.config import Config
from python.preprocess_data import create_proiel_cache


class Proiel(Dataset):
    def __init__(self, encodings: BatchEncoding, evaluate: bool = False):
        self.encodings: BatchEncoding = encodings

    def __len__(self):
        return len(self.encodings.data["input_ids"])

    def __getitem__(self, idx: int):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item


class Text:
    def __init__(self, content: str, indices: tuple[int, int] = None):
        self.content: str = content
        self.indices: tuple[int, int] = indices


class TextPair:
    def __init__(self, text_1: Text, text_2: Text, cosine_similarity: float = 0):
        self.text_1: Text = text_1
        self.text_2: Text = text_2
        self.cosine_similarity: float = cosine_similarity

    def get_texts(self) -> list[Text]:
        return [self.text_1, self.text_2]


class ProielData:
    text_pairs: list[TextPair] = [
        TextPair(text_1=Text(content="tolle lectum tuum et vade in domum tuam", indices=(2, 2)),
                 text_2=Text(content="tolle grabattum tuum et vade in domum tuam", indices=(2, 2))),
        TextPair(text_1=Text(content="qui autem minor est in regno caelorum maior est illo", indices=(7, 7)),
                 text_2=Text(content="qui autem minor est in regno Dei maior est illo", indices=(7, 7))),
        TextPair(text_1=Text(content="et ait pueris suis", indices=(3, 3)),
                 text_2=Text(content="et ait discipulis suis", indices=(3, 3))),
        TextPair(text_1=Text(content="interrogabo vos et ego unum sermonem", indices=(7, 7)),
                 text_2=Text(content="interrogabo vos et ego unum verbum", indices=(7, 7))),
        TextPair(text_1=Text(content="cuius est imago haec et suprascriptio", indices=(6, 10)),
                 text_2=Text(content="cuius est imago haec et inscriptio", indices=(6, 7))),
        TextPair(text_1=Text(content="respondens autem Iesus ait illis", indices=(3, 3)),
                 text_2=Text(content="respondens autem praeses ait illis", indices=(3, 3))),
        TextPair(text_1=Text(content="sexta feria in Syon", indices=(3, 3)),
                 text_2=Text(content="sexta die in Syon", indices=(3, 3))),
        TextPair(text_1=Text(content="et ecce vox de caelis dicens", indices=(5, 5)),
                 text_2=Text(content="et ecce vox de nube dicens", indices=(5, 5))),
        TextPair(text_1=Text(content="nonne et publicani hoc faciunt", indices=(3, 3)),
                 text_2=Text(content="nonne et ethnici hoc faciunt", indices=(3, 5))),
        TextPair(text_1=Text(content="et ait illi Iesus", indices=(4, 4)),
                 text_2=Text(content="et ait illi scriba", indices=(4, 4))),
        TextPair(text_1=Text(content="et respondens centurio ait", indices=(3, 3)),
                 text_2=Text(content="et respondens eis ait", indices=(3, 3))),
        TextPair(text_1=Text(content="vulpes foveas habent et volucres caeli tabernacula", indices=(10, 11)),
                 text_2=Text(content="vulpes foveas habent et volucres caeli nidos", indices=(10, 11))),
    ]


def calculate_cosine_similarity(model: OPTForCausalLM, tokenizer: GPT2TokenizerFast,
                                text_pairs: list[TextPair], do_print: bool = False) -> list[TextPair]:
    """ Calculates cosine similarity for pairs of texts. """
    for tp in text_pairs:
        seg_vec_1: torch.Tensor = get_embedding(tp.text_1.content, model, tokenizer, tp.text_1.indices)
        seg_vec_2: torch.Tensor = get_embedding(tp.text_2.content, model, tokenizer, tp.text_2.indices)
        cosine_similarity: torch.Tensor = torch.cosine_similarity(seg_vec_1, seg_vec_2, dim=0)
        cosine_float = float(cosine_similarity)
        if do_print:
            print(f"{round(cosine_float, 5)} || {tp.text_1.content} ; {tp.text_2.content}")
        tp.cosine_similarity = cosine_float
    return text_pairs


def correlate_cosine_similarity_with_text_length(model: OPTForCausalLM, tokenizer: GPT2TokenizerFast):
    """ Checks how much cosine similarity depends on text length. """

    class Helper:
        segment_lengths: list[int] = []
        segments: list[str] = []
        text_length_to_cosine: dict[float, list[float]] = dict()
        type_to_segment_ids: dict[str, set[int]] = dict()
        type_to_examples: dict[str, list[TextPair]] = dict()
        types: set[str] = set()

    def get_cosine_similarities(helper: Helper):
        """ Calculates text length and cosine similarity for all text pairs. """
        for key in tqdm(helper.type_to_examples):
            segment_to_idx: dict[str, tuple[int, int]] = get_token_indices(helper, key)
            for tp in helper.type_to_examples[key]:
                for txt in tp.get_texts():
                    txt.indices = segment_to_idx[txt.content]
            text_pairs: list[TextPair] = calculate_cosine_similarity(model, tokenizer, helper.type_to_examples[key])
            for tp in text_pairs:
                if tp.cosine_similarity == 1:
                    # skip, the text pairs are identical
                    continue
                text_length: float = sum([len(tp.text_1.content), len(tp.text_2.content)]) / 2
                if text_length not in helper.text_length_to_cosine:
                    helper.text_length_to_cosine[text_length] = []
                helper.text_length_to_cosine[text_length].append(tp.cosine_similarity)
        json.dump(helper.text_length_to_cosine, open(Config.cosine_similarity_vs_text_length_path, "w+"))

    def get_token_indices(helper: Helper, key: str) -> dict[str, tuple[int, int]]:
        """ Retrieves the indices of target tokens in a text sequence. """
        segment_to_idx: dict[str, tuple[int, int]] = dict()
        for text_pair in helper.type_to_examples[key]:
            for text_content in [text_pair.text_1.content, text_pair.text_2.content]:
                # get target token indices
                input_ids: list[int] = tokenizer.encode(text_content)
                # skip first token, it is the special EOS token </s>
                strings_by_id: list[str] = tokenizer.convert_ids_to_tokens(input_ids)[1:]
                tokens: list[str] = [strings_by_id[0]]
                ids: list[list[int]] = [[0]]
                # skip the first string, it is already present in the lists
                for i in range(1, len(strings_by_id)):
                    if strings_by_id[i][0] == "Ġ":
                        # new token, check the old one
                        if key in tokens[-1]:
                            break
                        tokens.append(strings_by_id[i])
                        ids.append([i])
                    else:
                        tokens[-1] += strings_by_id[i]
                        ids[-1].append(i)
                target_indices: tuple[int, int] = (ids[-1][0], ids[-1][-1])
                segment_to_idx[text_content] = target_indices
        return segment_to_idx

    def process_treebank() -> Helper:
        """ Extracts types and segments from a treebank. """
        print("processing treebank...")
        conllu_all: list[TokenList] = pickle.load(open(Config.proiel_pickle, "rb"))
        helper: Helper = Helper()
        for idx, sent in tqdm(enumerate(conllu_all)):
            tokens: list[str] = []
            for tok in sent.tokens:
                form: str = tok["form"]
                tokens.append(form)
                helper.types.add(form)
                if form not in helper.type_to_segment_ids:
                    helper.type_to_segment_ids[form] = set()
                helper.type_to_segment_ids[form].add(idx)
            helper.segments.append(" ".join(tokens))
            helper.segment_lengths.append(len(helper.segments[-1]))
        return helper

    def process_types(helper: Helper) -> Helper:
        """ Processes types by matching them with relevant segments of similar length. """
        print("processing types...")
        for item in tqdm(helper.types):
            if item not in helper.type_to_examples:
                helper.type_to_examples[item] = []
            segments_with_item_and_length: list[tuple[str, int]] = []
            for i in helper.type_to_segment_ids[item]:
                segments_with_item_and_length.append((helper.segments[i], helper.segment_lengths[i]))
            segments_with_item_and_length = sorted(segments_with_item_and_length, key=lambda x: x[1])
            segments_length: int = len(segments_with_item_and_length)
            for i in range(0, segments_length - 1, 2):
                helper.type_to_examples[item].append(
                    TextPair(text_1=Text(content=segments_with_item_and_length[i][0]),
                             text_2=Text(content=segments_with_item_and_length[i + 1][0])))
        return helper

    helper: Helper = process_treebank()
    helper = process_types(helper)
    # get_cosine_similarities(helper)
    json_dict: dict[str, list[float]] = json.load(open(Config.cosine_similarity_vs_text_length_path))
    helper.text_length_to_cosine = {float(x): json_dict[x] for x in json_dict}
    x: list[float] = sorted([float(z) for z in helper.text_length_to_cosine.keys()])
    y: list[float] = [float(np.average(helper.text_length_to_cosine[z])) for z in x]
    lr: LinearRegression = linear_regression(x, y)
    plt.scatter(x, y, s=9)
    y_regression: list[float] = [lr.intercept + (lr.slope * z) for z in x]
    plt.plot(x, y_regression, "-r")
    plt.xlabel("mean pairwise average of text length in characters")
    plt.ylabel("cosine similarity")
    plt.title("Correlation of Cosine Similarity and Text Length for Each Type in PROIEL")
    plt.savefig(os.path.join(Config.figures_dir, "cosine_similarity_vs_text_length.pdf"), dpi=600, format="pdf")
    plt.show()


def count_types_in_proiel():
    conllu_all: list[TokenList] = pickle.load(open(Config.proiel_pickle, "rb"))
    types: set[str] = set()
    for sent in conllu_all:
        for tok in sent.tokens:
            types.add(tok["form"])
    print(f"{len(types)} types overall")
    counter: Counter = Counter({x: len(x) for x in types})
    print(f"Longest types: {counter.most_common(5)}")


def generate(model: OPTForCausalLM, tokenizer: GPT2TokenizerFast):
    """ Generates text for a given input. """
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device: torch.device = torch.device("cpu")
    pl: Pipeline = pipeline("text-generation", model=model, device=device, tokenizer=tokenizer)
    txt: str = "Gallia omnis est divisa in partes tres"
    for x in pl(txt, num_return_sequences=5, do_sample=True):
        print(x["generated_text"])


def get_cosine_similarity_for_text_pairs(model: OPTForCausalLM, tokenizer: GPT2TokenizerFast) -> None:
    """ Calculates cosine similarity for pairs of texts and measures correlation with text length. """
    results: list[TextPair] = calculate_cosine_similarity(model, tokenizer, ProielData.text_pairs, True)
    results = sorted(results, key=lambda x: x.cosine_similarity, reverse=True)
    x: list[float] = [x.cosine_similarity for x in results]
    y: list[float] = [sum([len(x.text_1.content), len(x.text_2.content)]) / 2 for x in results]
    # lr: LinearRegression = linear_regression(x, y)
    # x_np: numpy.array = numpy.array(x)[:, None]
    # lr: object = LinearRegression().fit(x_np, y)
    # print(lr.coef_, lr.intercept_)
    correlation_r: np.array = np.corrcoef(x, y)
    print(correlation_r)


def get_embedding(
        input_text: str, model: OPTForCausalLM, tokenizer: GPT2TokenizerFast, indices: tuple[int, int],
        token_mode: bool = True) -> torch.Tensor:
    """ Calculates embeddings for segments of text from a given model and tokenizer. """
    be: BatchEncoding = tokenizer(input_text, return_tensors="pt").to(Config.device)
    # print(tokenizer.convert_ids_to_tokens(be.data["input_ids"][0])) # check how the embeddings map to tokens
    clmowp: CausalLMOutputWithPast = model(**be, output_hidden_states=True)
    hidden_states: tuple[torch.Tensor] = clmowp.hidden_states
    hidden_states_stacked: torch.Tensor = torch.stack(hidden_states)
    if token_mode:
        # use only the hidden states for the target token
        hidden_states_token: torch.Tensor = hidden_states_stacked[:, :, indices[0]:indices[1] + 1, :]
        # for tokens with multiple subordinate embeddings, average them to obtain the overall embedding
        hidden_states_token_merged: torch.Tensor = torch.mean(hidden_states_token, dim=2)
        # use the average of all hidden states (1 vector per layer) to obtain a flat vector (1 vector for all layers)
        token_vector: torch.Tensor = torch.mean(hidden_states_token_merged, dim=0)[0]
        # use all token embeddings because we cannot consistently map them to a single original token
        # segment_hidden_states: torch.Tensor = torch.mean(hidden_states_stacked, dim=2)
        # segment_vector: torch.Tensor = torch.mean(segment_hidden_states, dim=0)[0]
        return token_vector
    else:
        # we are in segment mode, use all tokens
        hidden_states_tokens: torch.Tensor = hidden_states_stacked[:, :, :, :]
        # TODO: calculate segment vector
        return torch.zeros(1)


def get_mean_proiel_segment_length():
    texts: list[str] = get_proiel_text()
    lengths: list[int] = [len(x) for x in texts]
    print(np.mean([len(x) for x in texts]))
    x: list[int] = list(set(lengths))
    plt.scatter(x, [lengths.count(z) for z in x], s=9)
    plt.xscale("log")
    plt.xlabel("text length in characters")
    plt.ylabel("number of segments")
    plt.title("Distribution of segment lengths in PROIEL")
    plt.savefig(os.path.join(Config.figures_dir, "segment_lengths_proiel.pdf"), dpi=600, format="pdf")
    plt.show()


def get_proiel_text() -> list[str]:
    """ Extracts all segments and tokens from PROIEL and returns plain text with linebreak segmentation. """
    conllu_all: list[TokenList] = pickle.load(open(Config.proiel_pickle, "rb"))
    texts: list[str] = [" ".join([y["form"] for y in sent.tokens]) for sent in conllu_all]
    return texts


def train_model(vocab_size: int, model_max_length: int):
    """ Trains a new language model with a given text corpus and tokenizer. """
    print(f"CUDA available: {torch.cuda.is_available()}")
    create_proiel_cache(Config.proiel_dir, Config.proiel_pickle)
    texts: list[str] = get_proiel_text()
    # train_tokenizer(vocab_size, model_max_length)
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(Config.tokenizer_dir)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = 0
    encodings: BatchEncoding = tokenizer(texts, truncation=True, padding=PaddingStrategy.LONGEST)
    model_config: OPTConfig = OPTConfig(
        hidden_size=384, ffn_dim=1536, num_hidden_layers=8, num_attention_heads=8, vocab_size=vocab_size,
        max_position_embeddings=model_max_length)  #
    model: OPTForCausalLM = OPTForCausalLM(model_config)
    print(f"{int(model.num_parameters()) // 1000000}M params")
    eval_steps: Union[int, float] = 0.1
    args: TrainingArguments = TrainingArguments(
        do_train=True, do_eval=True, per_device_train_batch_size=1, per_device_eval_batch_size=16,  # 8 32
        gradient_accumulation_steps=1, eval_accumulation_steps=1, num_train_epochs=1,  # 8 32
        evaluation_strategy=IntervalStrategy.STEPS, logging_strategy=IntervalStrategy.STEPS,
        save_strategy=IntervalStrategy.STEPS, logging_steps=eval_steps, eval_steps=eval_steps,
        save_steps=eval_steps,
        save_total_limit=3, no_cuda=False, gradient_checkpointing=True, output_dir=Config.output_dir,
        bf16=False, learning_rate=3e-4)  # True
    proiel_ds: Dataset = Proiel(encodings)
    train_ds, eval_ds = random_split(proiel_ds, [0.95, 0.05])
    data_collator: DataCollatorForLanguageModeling = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    trainer: Trainer = Trainer(model=model, args=args, callbacks=[TensorBoardCallback], train_dataset=train_ds,
                               eval_dataset=eval_ds, data_collator=data_collator)
    trainer.train()
    trainer.save_model(Config.model_dir)


def train_tokenizer(vocab_size: int, model_max_length: int):
    """ Trains a new tokenizer on a given text corpus. """
    texts: list[str] = get_proiel_text()
    # Customize training
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(
        "facebook/opt-350m", model_max_length=model_max_length)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer = tokenizer.train_new_from_iterator(
        text_iterator=texts, vocab_size=vocab_size, length=len(texts),
        new_special_tokens=["<s>", "</s>", "<unk>", "<mask>", ])
    # Save files to disk
    tokenizer.save_pretrained(Config.tokenizer_dir)


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    model: OPTForCausalLM = OPTForCausalLM.from_pretrained(Config.model_dir).to(Config.device)
    tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(Config.tokenizer_dir)
    vocab_size: int = 10000  # 32000 default for LLama
    model_max_length: int = 512
    # train_tokenizer(vocab_size=vocab_size, model_max_length=model_max_length)
    # train_model(vocab_size, model_max_length)
    # get_cosine_similarity_for_text_pairs(model, tokenizer)
    # generate(model, tokenizer)
    # correlate_cosine_similarity_with_text_length(model, tokenizer)
    get_mean_proiel_segment_length()
    a = 0

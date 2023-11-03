import os
import pickle
from collections import Counter
from typing import Union

import torch
from conllu import TokenList
from torch.utils.data import Dataset, random_split
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


def get_cosine_similarity_for_text_pairs(model: OPTForCausalLM, tokenizer: GPT2TokenizerFast):
    text_pairs: list[tuple[str, str, tuple[int, int], tuple[int, int]]] = [
        ("tolle lectum tuum et vade in domum tuam", "tolle grabattum tuum et vade in domum tuam", (2, 2), (2, 2)),
        ("qui autem minor est in regno caelorum maior est illo", "qui autem minor est in regno Dei maior est illo",
         (7, 7), (7, 7)),
        ("et ait pueris suis", "et ait discipulis suis", (3, 3), (3, 3)),
        ("interrogabo vos et ego unum sermonem", "interrogabo vos et ego unum verbum", (7, 7), (7, 7)),
        ("cuius est imago haec et suprascriptio", "cuius est imago haec et inscriptio", (6, 10), (6, 7)),
        ("respondens autem Iesus ait illis", "respondens autem praeses ait illis", (3, 3), (3, 3))]
    text_pairs = [
        ("sexta feria in Syon", "sexta die in Syon", (3, 3), (3, 3)),
        ("et ecce vox de caelis dicens", "et ecce vox de nube dicens", (5, 5), (5, 5)),
        ("nonne et publicani hoc faciunt", "nonne et ethnici hoc faciunt", (3, 3), (3, 5)),
        ("et ait illi Iesus", "et ait illi scriba", (4, 4), (4, 4)),
        ("et respondens centurio ait", "et respondens eis ait", (3, 3), (3, 3)),
        ("vulpes foveas habent et volucres caeli tabernacula", "vulpes foveas habent et volucres caeli nidos", (10, 11),
         (10, 11))
    ]
    for tp in text_pairs:
        seg_vec_1: torch.Tensor = get_segment_embedding(tp[0], model, tokenizer, tp[2])
        seg_vec_2: torch.Tensor = get_segment_embedding(tp[1], model, tokenizer, tp[3])
        cosine_similarity: torch.Tensor = torch.cosine_similarity(seg_vec_1, seg_vec_2, dim=0)
        print(f"{float(cosine_similarity)} || {tp[:2]}")


def get_proiel_text() -> list[str]:
    """ Extracts all segments and tokens from PROIEL and returns plain text with linebreak segmentation. """
    conllu_all: list[TokenList] = pickle.load(open(Config.proiel_pickle, "rb"))
    texts: list[str] = [" ".join([y["form"] for y in sent.tokens]) for sent in conllu_all]
    return texts


def get_segment_embedding(
        input_text: str, model: OPTForCausalLM, tokenizer: GPT2TokenizerFast, indices: tuple[int, int]):
    """ Calculates embeddings for segments of text from a given model and tokenizer. """
    be: BatchEncoding = tokenizer(input_text, return_tensors="pt")
    # print(tokenizer.convert_ids_to_tokens(be.data["input_ids"][0]))  # check how the embeddings map to tokens
    clmowp: CausalLMOutputWithPast = model(**be, output_hidden_states=True)
    hidden_states: tuple[torch.Tensor] = clmowp.hidden_states
    hidden_states_stacked: torch.Tensor = torch.stack(hidden_states)
    # use only the hidden states for the target token
    hidden_states_token: torch.Tensor = hidden_states_stacked[:, :, indices[0]:indices[1] + 1, :]
    # for tokens with multiple subordinate embeddings, average them to obtain the overall embedding
    hidden_states_token_merged: torch.Tensor = torch.mean(hidden_states_token, dim=2)
    # use the average of all hidden states to obtain a flat vector
    token_vector: torch.Tensor = torch.mean(hidden_states_token_merged, dim=0)[0]
    # use all token embeddings because we cannot consistently map them to a single original token
    # segment_hidden_states: torch.Tensor = torch.mean(hidden_states_stacked, dim=2)
    # segment_vector: torch.Tensor = torch.mean(segment_hidden_states, dim=0)[0]
    return token_vector


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
        save_strategy=IntervalStrategy.STEPS, logging_steps=eval_steps, eval_steps=eval_steps, save_steps=eval_steps,
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


model: OPTForCausalLM = OPTForCausalLM.from_pretrained(Config.model_dir)
tokenizer: GPT2TokenizerFast = GPT2TokenizerFast.from_pretrained(Config.tokenizer_dir)
vocab_size: int = 10000  # 32000 default for LLama
model_max_length: int = 512
# train_tokenizer(vocab_size=vocab_size, model_max_length=model_max_length)
# train_model(vocab_size, model_max_length)
get_cosine_similarity_for_text_pairs(model, tokenizer)
# generate(model, tokenizer)
a = 0

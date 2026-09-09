import os.path

import torch


class Config:
    output_dir: str = os.path.abspath("output")
    asrael_dir: str = os.path.abspath("..")
    cosine_similarity_vs_text_length_path: str = "text_length_to_cosine.json"
    data_dir: str = os.path.join(os.path.abspath(".."), "data")
    device: torch.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    figures_dir: str = os.path.join(asrael_dir, "figures")
    model_dir: str = os.path.join(output_dir, "model")
    proiel_dir: str = os.path.join(data_dir, "proiel_conllu")
    proiel_pickle: str = os.path.join(data_dir, "proiel_conllu.pickle")
    tokenizer_dir: str = os.path.abspath("opt_latin_hf_tokenizer")

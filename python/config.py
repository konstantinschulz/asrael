import os.path


class Config:
    output_dir: str = os.path.abspath("output")
    data_dir: str = os.path.join(os.path.abspath(".."), "data")
    model_dir: str = os.path.join(output_dir, "model")
    proiel_dir: str = os.path.join(data_dir, "proiel_conllu")
    proiel_pickle: str = os.path.join(data_dir, "proiel_conllu.pickle")
    tokenizer_dir: str = os.path.abspath("opt_latin_hf_tokenizer")

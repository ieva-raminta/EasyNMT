from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from typing import List
import logging
# import sys
# sys.path.append("../../") # Adds higher directory to python modules path.
# from debias_files.src.debias_manager import DebiasManager
# from debias_files.src.consts import LANGUAGE_STR_TO_INT_MAP
import copy

logger = logging.getLogger(__name__)


class AutoModel:
    def __init__(self, model_name: str, tokenizer_name: str = None, easynmt_path: str = None, lang_map=None, tokenizer_args=None, src_lang=None,tgt_lang=None):
        if tokenizer_args is None:
            tokenizer_args = {}
        tokenizer_args["src_lang"] =src_lang
        tokenizer_args["tgt_lang"] =tgt_lang
        if lang_map is None:
            lang_map = {}

        if tokenizer_name is None:
            tokenizer_name = model_name

        self.lang_map = lang_map
        self.tokenizer_args = tokenizer_args

        if model_name == ".":
            model_name = easynmt_path

        if tokenizer_name == ".":
            tokenizer_name = easynmt_path

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        print("tokenizer_name")
        print(tokenizer_name)
        print("**self.tokenizer_args")
        print(self.tokenizer_args)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **self.tokenizer_args)
        self.max_length = None
    # def _sanity_check_debias(self, orig_embeddings, debiased_embeddings,model):
    #     c=0
    #     orig_embeddings = orig_embeddings.to("cuda:0")
    #     debiased_embeddings = debiased_embeddings.to("cuda:0")
    #     for i in range(len(orig_embeddings)):
    #         if (debiased_embeddings[i] != orig_embeddings[i]).any():
    #             c += 1
    #     if model.debias_target_language:
    #         if model.target_lang == 'he':
    #             assert (c == len(model.hebrew_professions))
    #         elif model.target_lang == 'de':
    #             assert (c == len(model.german_professions))
    #         # elif model.target_lang == 'ru':
    #         #     assert (c == len(model.russian_professions))
    #     else:
    #         assert (c == len(model.professions))


    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str, device: str, beam_size: int = 5, **kwargs):

        self.model.to(device)

        if source_lang in self.lang_map:
            source_lang = self.lang_map[source_lang]

        if target_lang in self.lang_map:
            target_lang = self.lang_map[target_lang]

        self.tokenizer.src_lang = source_lang
        inputs = self.tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(device)
        bad=[[22008], [77848], [195812], [43373], [56050], [17496], [160675], [45733], [172647], [135969], [51517], [184427], [117914], [110309], [185118], [9836], [93324], [37896], [21861], [115835], [121399], [148665], [166513], [101785], [31095], [183093], [106001], [219600], [29041], [171751], [91519], [70035], [27941], [30391], [22072], [53294], [196592], [185256], [160020], [195644], [188183], [133604], [107653], [23282]]
        # bad=[[]]
        with torch.no_grad():
            if hasattr(self.tokenizer, 'lang_code_to_id'):
                kwargs['forced_bos_token_id'] = self.tokenizer.lang_code_to_id[target_lang]
            translated = self.model.generate(**inputs, num_beams=beam_size, bad_words_ids=bad,**kwargs)
            output = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return output

    def save(self, output_path):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        return {
            "model_name": ".",
            "tokenizer_name": ".",
            "lang_map": self.lang_map,
            "tokenizer_args": self.tokenizer_args
        }

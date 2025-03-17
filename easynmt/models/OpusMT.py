import copy
import time
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List
import logging
import sys
# sys.path.append("..") # Adds higher directory to python modules path.
sys.path.append("../../") # Adds higher directory to python modules path.
from debias_manager import DebiasManager
from consts import LANGUAGE_STR_TO_INT_MAP
from datetime import datetime
logger = logging.getLogger(__name__)
from torch.nn.functional import log_softmax
import os 

class OpusMT:
    def __init__(self, easynmt_path: str = None, max_loaded_models: int = 10):
        self.models = {}
        self.max_loaded_models = max_loaded_models
        self.max_length = None
    def _sanity_check_debias(self, orig_embeddings, debiased_embeddings,model):
        c=0
        for i in range(len(orig_embeddings)):
            if (debiased_embeddings[i] != orig_embeddings[i]).any():
                c += 1
        if model.debias_target_language:
            if model.target_lang == 'he':
                assert (c == len(model.hebrew_professions))
            elif model.target_lang == 'de':
                assert (c == len(model.german_professions))
            elif model.target_lang == 'ru':
                assert (c == len(model.russian_professions))
            elif model.target_lang == 'fr':
                assert (c == len(model.french_professions))
            elif model.target_lang == 'es':
                assert (c == len(model.spanish_professions
                                ))
        else:
            assert (c == len(model.professions))
    def load_model(self, model_name, **kwargs):
        if model_name in self.models:
            self.models[model_name]['last_loaded'] = time.time()
            return self.models[model_name]['tokenizer'], self.models[model_name]['model']
        else:
            logger.info("Load model: "+model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            ### comment: step 4
            model = MarianMTModel.from_pretrained(model_name)#, **kwargs)
            ### comment: sanity check
            # config = "{'USE_DEBIASED': 0, 'LANGUAGE': "+str(LANGUAGE_STR_TO_INT_MAP[tokenizer.target_lang])+", 'COLLECT_EMBEDDING_TABLE': 0, 'DEBIAS_METHOD': 0, 'TRANSLATION_MODEL': 1}"
            # debiasManager = DebiasManager(config)
            # debiasManager.prepare_data_to_debias(tokenizer.get_vocab(), model.get_input_embeddings().weight.data)
            # debiasManager.sanity_check_origin_embedding( tokenizer.get_vocab(), model.get_input_embeddings().weight.data)

            if kwargs['use_debiased']:
                print("using debiased embeddings")
                target_lang='es' if tokenizer.target_lang=='spa' else tokenizer.target_lang
                config_str = "{'USE_DEBIASED': 1" \
                             ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) +\
                             ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                             ", 'TRANSLATION_MODEL': " + str(kwargs['translation_model']) + \
                             ", 'DEBIAS_ENCODER': " + str(kwargs['debias_encoder']) + "" \
                             ", 'BEGINNING_DECODER_DEBIAS': " + str(kwargs['beginning_decoder_debias']) + \
                             ", 'END_DECODER_DEBIAS': " + str(kwargs['end_decoder_debias']) +\
                             ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                dict = model.state_dict()
                # option 1: debias encoder inputs
                if kwargs['debias_encoder']:
                    print('debias_encoder')
                    config_str = "{'USE_DEBIASED': 1" \
                                 ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                                 ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                                 ", 'TRANSLATION_MODEL': " + str(kwargs['translation_model']) + \
                                 ", 'DEBIAS_ENCODER': 1" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                                 ", 'END_DECODER_DEBIAS': 0"+ \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_encoder = copy.deepcopy(dict['model.encoder.embed_tokens.weight'])
                    debias_manager_encoder = DebiasManager.get_manager_instance(config_str, weights_encoder, tokenizer)
                    new_embeddings_encoder = torch.from_numpy(debias_manager_encoder.debias_embedding_table())
                    self._sanity_check_debias(weights_encoder, new_embeddings_encoder,debias_manager_encoder)
                    dict['model.encoder.embed_tokens.weight'] = new_embeddings_encoder
                # option 2: debias decoder inputs
                if kwargs['beginning_decoder_debias']:
                    print('beginning_decoder_debias')
                    config_str = "{'USE_DEBIASED': 1" \
                                 ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                                 ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                                 ", 'TRANSLATION_MODEL': " + str(kwargs['translation_model']) + \
                                 ", 'DEBIAS_ENCODER': 0" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 1" + \
                                 ", 'END_DECODER_DEBIAS': 0" + \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_decoder = copy.deepcopy(dict['model.decoder.embed_tokens.weight'])
                    debias_manager_decoder = DebiasManager.get_manager_instance(config_str, weights_decoder, tokenizer, debias_target_language=True)
                    new_embeddings_decoder = torch.from_numpy(debias_manager_decoder.debias_embedding_table())
                    self._sanity_check_debias(weights_decoder, new_embeddings_decoder,debias_manager_decoder)
                    dict['model.decoder.embed_tokens.weight'] = new_embeddings_decoder

                # # option 3: debias decoder outputs
                if kwargs['end_decoder_debias']:
                    print('end_decoder_debias')
                    config_str = "{'USE_DEBIASED': 1" \
                                 ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                                 ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                                 ", 'TRANSLATION_MODEL': 1" \
                                 ", 'DEBIAS_ENCODER': 0" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                                 ", 'END_DECODER_DEBIAS': 1" + \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_decoder_outputs = copy.deepcopy(dict['lm_head.weight'])
                    debias_manager_decoder_outputs = DebiasManager.get_manager_instance(config_str, weights_decoder_outputs, tokenizer, debias_target_language=True)
                    new_embeddings_decoder_outputs = torch.from_numpy(debias_manager_decoder_outputs.debias_embedding_table())
                    self._sanity_check_debias(weights_decoder_outputs, new_embeddings_decoder_outputs,debias_manager_decoder_outputs)
                    dict['lm_head.weight'] = new_embeddings_decoder_outputs

                model.load_state_dict(dict)
                print("Saving model...")
                model.save_pretrained(f"/home/irs38/intrinsic-debiasing-performance-on-NMT/EasyNMT/models/models/en-{target_lang}")
                tokenizer.save_pretrained(f"/home/irs38/intrinsic-debiasing-performance-on-NMT/EasyNMT/models/tokenizers/en-{target_lang}")

                # dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
                # model.save_pretrained(f"/cs/usr/bareluz/gabi_labs/nematus_clean/models/models/en-{target_lang}-"
                #                       f"DEBIAS_METHOD-{str(kwargs['debias_method'])}-"
                #                       f"DEBIAS_ENCODER-{str(kwargs['debias_encoder'])}-"
                #                       f"BEGINNING_DECODER_DEBIAS-{str(kwargs['beginning_decoder_debias'])}-"
                #                       f"END_DECODER_DEBIAS-{str(kwargs['end_decoder_debias'])}-"
                #                       f"WORDS_TO_DEBIAS-{str(kwargs['words_to_debias'])}---{dt_string}")
                # tokenizer.save_pretrained(f"/cs/usr/bareluz/gabi_labs/nematus_clean/models/tokenizers/en-{target_lang}-"
                #                       f"DEBIAS_METHOD-{str(kwargs['debias_method'])}-"
                #                       f"DEBIAS_ENCODER-{str(kwargs['debias_encoder'])}-"
                #                       f"BEGINNING_DECODER_DEBIAS-{str(kwargs['beginning_decoder_debias'])}-"
                #                       f"END_DECODER_DEBIAS-{str(kwargs['end_decoder_debias'])}-"
                #                       f"WORDS_TO_DEBIAS-{str(kwargs['words_to_debias'])}---{dt_string}")

            else:
                print("using non debiased embeddings")

            model.eval()

            if len(self.models) >= self.max_loaded_models:
                oldest_time = time.time()
                oldest_model = None
                for loaded_model_name in self.models:
                    if self.models[loaded_model_name]['last_loaded'] <= oldest_time:
                        oldest_model = loaded_model_name
                        oldest_time = self.models[loaded_model_name]['last_loaded']
                del self.models[oldest_model]

            self.models[model_name] = {'tokenizer': tokenizer, 'model': model, 'last_loaded': time.time()}
            return tokenizer, model

    def translate_sentences(self, sentences: List[str], source_lang: str, target_lang: str, device: str, beam_size: int = 5, **kwargs):
        model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(source_lang, target_lang)
        ### comment: step 3
        tokenizer, model = self.load_model(model_name,**kwargs)
        model.to(device)
        inputs = tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        names = "names"

        for key in inputs:
            inputs[key] = inputs[key].to(device)
        
        debiased = 'debiased' if kwargs['use_debiased'] else 'nondebiased'

        outputs = model.generate(
            **inputs,
            num_return_sequences=128,
            num_beams=1,
            do_sample=True,
            temperature=1,
            epsilon_cutoff=0.2,
            output_scores=True, 
            return_dict_in_generate=True
        )

        sequences = outputs.sequences
        scores = outputs.scores        

        log_probs = []
        for seq_idx, sequence in enumerate(sequences):
            seq_log_prob = 0.0  
            for t, step_scores in enumerate(scores):
                log_probs_t = log_softmax(step_scores, dim=-1)
                token_id = sequence[t+1] 
                if token_id not in tokenizer.all_special_ids:
                    seq_log_prob += log_probs_t[seq_idx, token_id].clamp(min=-1e10).item()

            log_probs.append(seq_log_prob)

        with open(f'/home/irs38/uncertainty/translations/samples_{debiased}-{model_name.split("/")[-1]}_{target_lang}_temp1.0_{names}_unambiguous.txt', 'a') as f:
            for sample in sequences:
                decoded = tokenizer.decode(sample, skip_special_tokens=True)
                f.write(decoded + '\n')
        with open(f'/home/irs38/uncertainty/translations/log_probs_{debiased}-{model_name.split("/")[-1]}_{target_lang}_temp1.0_{names}_unambiguous.txt', 'a') as f:
            for log_prob in log_probs:
                f.write(str(log_prob) + '\n')

        kwargs.pop('use_debiased')
        kwargs.pop('debias_method')
        kwargs.pop('debias_encoder')
        kwargs.pop('beginning_decoder_debias')
        kwargs.pop('end_decoder_debias')
        kwargs.pop('words_to_debias')
        kwargs.pop('translation_model')

        return outputs

    def save(self, output_path):
        return {"max_loaded_models": self.max_loaded_models}


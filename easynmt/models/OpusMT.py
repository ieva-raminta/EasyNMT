import time
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List
import logging
from debias_files.debias_manager import DebiasManager
from debias_files.consts import LANGUAGE_STR_TO_INT_MAP
logger = logging.getLogger(__name__)


class OpusMT:
    def __init__(self, easynmt_path: str = None, max_loaded_models: int = 10):
        self.models = {}
        self.max_loaded_models = max_loaded_models
        self.max_length = None

    def load_model(self, model_name, **kwargs):
        if model_name in self.models:
            self.models[model_name]['last_loaded'] = time.time()
            return self.models[model_name]['tokenizer'], self.models[model_name]['model']
        else:
            logger.info("Load model: "+model_name)
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            ### comment: step 4
            model = MarianMTModel.from_pretrained(model_name,tokenizer= tokenizer, **kwargs)
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
                             ", 'TRANSLATION_MODEL': 1" \
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
                                 ", 'TRANSLATION_MODEL': 1" \
                                 ", 'DEBIAS_ENCODER': 1" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                                 ", 'END_DECODER_DEBIAS': 0"+ \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_encoder = dict['model.encoder.embed_tokens.weight']
                    debias_manager_encoder = DebiasManager.get_manager_instance(config_str, weights_encoder, tokenizer)
                    new_embeddings_encoder = torch.from_numpy(debias_manager_encoder.debias_embedding_table())
                    dict['model.encoder.embed_tokens.weight'] = new_embeddings_encoder
                # option 2: debias decoder inputs
                if kwargs['beginning_decoder_debias']:
                    print('beginning_decoder_debias')
                    config_str = "{'USE_DEBIASED': 1" \
                                 ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                                 ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                                 ", 'TRANSLATION_MODEL': 1" \
                                 ", 'DEBIAS_ENCODER': 0" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 1" + \
                                 ", 'END_DECODER_DEBIAS': 0" + \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_decoder = dict['model.decoder.embed_tokens.weight']
                    debias_manager_decoder = DebiasManager.get_manager_instance(config_str, weights_decoder, tokenizer, debias_target_language=True)
                    new_embeddings_decoder = torch.from_numpy(debias_manager_decoder.debias_embedding_table())
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
                    weights_decoder_outputs = dict['lm_head.weight']
                    debias_manager_decoder_outputs = DebiasManager.get_manager_instance(config_str, weights_decoder_outputs, tokenizer, debias_target_language=True)
                    new_embeddings_decoder_outputs = torch.from_numpy(debias_manager_decoder_outputs.debias_embedding_table())
                    dict['lm_head.weight'] = new_embeddings_decoder_outputs

                model.load_state_dict(dict)
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
        tokenizer, model = self.load_model(model_name)
        model.to(device)

        inputs = tokenizer(sentences, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].to(device)

        with torch.no_grad():
            kwargs.pop('use_debiased')
            kwargs.pop('debias_method')
            kwargs.pop('debias_encoder')
            kwargs.pop('beginning_decoder_debias')
            kwargs.pop('end_decoder_debias')
            kwargs.pop('words_to_debias')
            translated = model.generate(**inputs, num_beams=beam_size, **kwargs)
            output = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

        return output

    def save(self, output_path):
        return {"max_loaded_models": self.max_loaded_models}


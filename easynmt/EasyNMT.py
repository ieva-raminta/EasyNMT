import os
import torch
from .util import http_get, import_from_string, fullname
import json
from . import __DOWNLOAD_SERVER__
from typing import List, Union, Dict, FrozenSet, Set, Iterable
import numpy as np
import tqdm
import nltk
import torch.multiprocessing as mp
import queue
import math
import re
import logging
import time
import os
import copy
import sys
sys.path.append("../../") # Adds higher directory to python modules path.
from .debias_manager import DebiasManager
from .consts import LANGUAGE_STR_TO_INT_MAP, TranslationModelsEnum
logger = logging.getLogger(__name__)

class EasyNMT:
    def __init__(self, model_name: str = None, cache_folder: str = None, translator=None, load_translator: bool = True, device=None, max_length: int = None, **kwargs):
        """
        Easy-to-use, state-of-the-art machine translation
        :param model_name:  Model name (see Readme for available models)
        :param cache_folder: Which folder should be used for caching models. Can also be set via the EASYNMT_CACHE env. variable
        :param translator: Translator object. Set to None, to automatically load the model via the model name.
        :param load_translator: If set to false, it will only load the config but not the translation engine
        :param device: CPU / GPU device for PyTorch
        :param max_length: Max number of token per sentence for translation. Longer text will be truncated
        :param kwargs: Further optional parameters for the different models
        """
        self._model_name = model_name
        self._fasttext_lang_id = None
        self._lang_detectors = [self.language_detection_fasttext, self.language_detection_langid, self.language_detection_langdetect]
        self._lang_pairs = frozenset()

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.config = None

        if cache_folder is None:
            if 'EASYNMT_CACHE' in os.environ:
                cache_folder = os.environ['EASYNMT_CACHE']
            else:
                cache_folder = os.path.join(torch.hub._get_torch_home(), 'easynmt_v2')
        self._cache_folder = cache_folder

        if translator is not None:
            self.translator = translator
        else:
            if os.path.exists(model_name) and os.path.isdir(model_name):
                model_path = model_name
            else:
                model_name = model_name.lower()
                model_path = os.path.join(cache_folder, model_name)

                if not os.path.exists(model_path) or not os.listdir(model_path):
                    logger.info("Downloading EasyNMT model {} and saving it at {}".format(model_name, model_path))

                    model_path_tmp = model_path.rstrip("/").rstrip("\\") + "_part"
                    os.makedirs(model_path_tmp, exist_ok=True)

                    #Download easynmt.json
                    config_url = __DOWNLOAD_SERVER__+"/{}/easynmt.json".format(model_name)
                    config_path = os.path.join(model_path_tmp, 'easynmt.json')
                    http_get(config_url, config_path)

                    with open(config_path) as fIn:
                        downloaded_config = json.load(fIn)

                    if 'files' in downloaded_config:
                        for filename, url in downloaded_config['files'].items():
                            logger.info("Download {} from {}".format(filename, url))
                            http_get(url, os.path.join(model_path_tmp, filename))

                    ##Rename tmp path
                    try:
                        os.rename(model_path_tmp, model_path)
                    except Exception:
                        pass

            with open(os.path.join(model_path, 'easynmt.json')) as fIn:
                self.config = json.load(fIn)

            if 'lang_pairs' in self.config:
                self._lang_pairs = frozenset(self.config['lang_pairs'])

            if load_translator:
                module_class = import_from_string(self.config['model_class'])
                if 'src_lang' in self.config['model_args'] and 'tgt_lang' in self.config['model_args']:
                    self.config['model_args']['src_lang'] = kwargs['src_lang']
                    self.config['model_args']['tgt_lang'] = kwargs['tgt_lang']
                elif 'src_lang' in self.config['model_args']['tokenizer_args'] and 'tgt_lang' in self.config['model_args']['tokenizer_args']:
                    self.config['model_args']['tokenizer_args']['src_lang'] = kwargs['src_lang']
                    self.config['model_args']['tokenizer_args']['tgt_lang'] = kwargs['tgt_lang']

                self.translator = module_class(easynmt_path=model_path, **self.config['model_args'], src_lang = kwargs['src_lang'],tgt_lang= kwargs['tgt_lang'])
                self.translator.max_length = max_length



    def _sanity_check_debias(self, orig_embeddings, debiased_embeddings,model):
        c=0
        orig_embeddings = orig_embeddings.to("cuda:0")
        debiased_embeddings = debiased_embeddings.to("cuda:0")
        for i in range(len(orig_embeddings)):
            if (debiased_embeddings[i] != orig_embeddings[i]).any():
                c += 1
        if model.debias_target_language:
            if model.target_lang == 'he':
                assert (c == len(model.hebrew_professions))
            elif model.target_lang == 'de':
                assert (c == len(model.german_professions))
            # elif model.target_lang == 'ru':
            #     assert (c == len(model.russian_professions))
        else:
            assert (c == len(model.professions))

    def translate(self, documents: Union[str, List[str]], target_lang: str, source_lang: str = None,
                  show_progress_bar: bool = False, beam_size: int = 5, batch_size: int = 16,
                  perform_sentence_splitting: bool = True, paragraph_split: str = "\n", sentence_splitter=None,  document_language_detection: bool = True,
                  **kwargs):
        """
        This method translates the given set of documents
        :param documents: If documents is a string, returns the translated document as string. If documents is a list of strings, translates all documents and returns a list.
        :param target_lang: Target language for the translation
        :param source_lang: Source language for all documents. If None, determines the source languages automatically.
        :param show_progress_bar: If true, plot a progress bar on the progress for the translation
        :param beam_size: Size for beam search
        :param batch_size: Number of sentences to translate at the same time
        :param perform_sentence_splitting: Longer documents are broken down sentences, which are translated individually
        :param paragraph_split: Split symbol for paragraphs. No sentences can go across the paragraph_split symbol.
        :param sentence_splitter: Method used to split sentences. If None, uses the default self.sentence_splitting method
        :param document_language_detection: Perform language detection on document level
        :param kwargs: Optional arguments for the translator model
        :return: Returns a string or a list of string with the translated documents
        """
        if kwargs['translation_model'] == TranslationModelsEnum.MBART50.value:
            #Method_args will store all passed arguments to method
            if kwargs['use_debiased']:
                print("using debiased embeddings")
                dict = self.translator.model.state_dict()
                # option 1: debias encoder inputs

                if kwargs['debias_encoder']:
                    print('debias_encoder') #todo add translaion_model to kwargs, it was overriten
                    config_str = "{'USE_DEBIASED': 1" \
                                 ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                                 ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                                 ", 'TRANSLATION_MODEL': " + str(kwargs['translation_model']) + \
                                 ", 'DEBIAS_ENCODER': 1" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                                 ", 'END_DECODER_DEBIAS': 0"+ \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_encoder = copy.deepcopy(dict['model.encoder.embed_tokens.weight'])
                    debias_manager_encoder = DebiasManager.get_manager_instance(config_str, weights_encoder, self.translator.tokenizer)
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
                    debias_manager_decoder = DebiasManager.get_manager_instance(config_str, weights_decoder, self.translator.tokenizer, debias_target_language=True)
                    new_embeddings_decoder = torch.from_numpy(debias_manager_decoder.debias_embedding_table())
                    self._sanity_check_debias(weights_decoder, new_embeddings_decoder,debias_manager_decoder)
                    dict['model.decoder.embed_tokens.weight'] = new_embeddings_decoder

                # # option 3: debias decoder outputs
                if kwargs['end_decoder_debias']:
                    print('end_decoder_debias')
                    config_str = "{'USE_DEBIASED': 1" \
                                 ", 'LANGUAGE': " + str(LANGUAGE_STR_TO_INT_MAP[target_lang]) + \
                                 ", 'DEBIAS_METHOD': " + str(kwargs['debias_method']) + \
                                 ", 'TRANSLATION_MODEL': " + str(kwargs['translation_model']) + \
                                 ", 'DEBIAS_ENCODER': 0" \
                                 ", 'BEGINNING_DECODER_DEBIAS': 0" + \
                                 ", 'END_DECODER_DEBIAS': 1" + \
                                 ", 'WORDS_TO_DEBIAS': " + str(kwargs['words_to_debias']) + "}"
                    weights_decoder_outputs = copy.deepcopy(dict['lm_head.weight'])
                    debias_manager_decoder_outputs = DebiasManager.get_manager_instance(config_str, weights_decoder_outputs, self.translator.tokenizer, debias_target_language=True)
                    new_embeddings_decoder_outputs = torch.from_numpy(debias_manager_decoder_outputs.debias_embedding_table())
                    self._sanity_check_debias(weights_decoder_outputs, new_embeddings_decoder_outputs,debias_manager_decoder_outputs)
                    dict['lm_head.weight'] = new_embeddings_decoder_outputs

                self.translator.model.load_state_dict(dict)
                # now = datetime.now()
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

            self.translator.model.eval()
            kwargs.pop('use_debiased')
            kwargs.pop('debias_method')
            kwargs.pop('debias_encoder')
            kwargs.pop('beginning_decoder_debias')
            kwargs.pop('end_decoder_debias')
            kwargs.pop('words_to_debias')
            kwargs.pop('translation_model')
        method_args = locals()
        del method_args['self']
        del method_args['kwargs']
        method_args.update(kwargs)
        if source_lang == target_lang:
            return documents

        is_single_doc = False
        if isinstance(documents, str):
            documents = [documents]
            is_single_doc = True

        if source_lang is None and document_language_detection:
            src_langs = [self.language_detection(doc) for doc in documents]

            # Group by languages
            lang2id = {}
            for idx, lng in enumerate(src_langs):
                if lng not in lang2id:
                    lang2id[lng] = []
                lang2id[lng].append(idx)

            # Translate language wise
            output = [None] * len(documents)
            for lng, ids in lang2id.items():
                logger.info("Translate documents of language: {}".format(lng))
                try:
                    method_args['documents'] = [documents[idx] for idx in ids]
                    method_args['source_lang'] = lng
                    translated = self.translate(**method_args)
                    for idx, translated_sentences in zip(ids, translated):
                        output[idx] = translated_sentences
                except Exception as e:
                    logger.warning("Exception: "+str(e))
                    raise e

            if is_single_doc and len(output) == 1:
                output = output[0]

            return output


        if perform_sentence_splitting:


            if sentence_splitter is None:
                sentence_splitter = self.sentence_splitting

            # Split document into sentences
            start_time = time.time()
            splitted_sentences = []
            sent2doc = []
            for doc in documents:
                paragraphs = doc.split(paragraph_split) if paragraph_split is not None else [doc]
                for para in paragraphs:
                    for sent in sentence_splitter(para.strip(), source_lang):
                        sent = sent.strip()
                        if len(sent) > 0:
                            splitted_sentences.append(sent)
                sent2doc.append(len(splitted_sentences))
            #logger.info("Sentence splitting done after: {:.2f} sec".format(time.time() - start_time))
            #logger.info("Translate {} sentences".format(len(splitted_sentences)))
            ### comment: step 1

            translated_sentences = self.translate_sentences(splitted_sentences, target_lang=target_lang, source_lang=source_lang, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size, **kwargs)

            # Merge sentences back to documents
            start_time = time.time()
            translated_docs = []
            for doc_idx in range(len(documents)):
                start_idx = sent2doc[doc_idx - 1] if doc_idx > 0 else 0
                end_idx = sent2doc[doc_idx]
                translated_docs.append(self._reconstruct_document(documents[doc_idx], splitted_sentences[start_idx:end_idx], translated_sentences[start_idx:end_idx]))

            #logger.info("Document reconstruction done after: {:.2f} sec".format(time.time() - start_time))
        else:

            translated_docs = self.translate_sentences(documents, target_lang=target_lang, source_lang=source_lang, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size, **kwargs)

        if is_single_doc:

            translated_docs = translated_docs[0]
        return translated_docs

    @staticmethod
    def _reconstruct_document(doc, org_sent, translated_sent):
        """
        This method reconstructs the translated document and
        keeps white space in the beginning / at the end of sentences.
        """
        sent_idx = 0
        char_idx = 0
        translated_doc = ""
        while char_idx < len(doc):
            if sent_idx < len(org_sent) and doc[char_idx] == org_sent[sent_idx][0]:
                translated_doc += translated_sent[sent_idx]
                char_idx += len(org_sent[sent_idx])
                sent_idx += 1
            else:
                translated_doc += doc[char_idx]
                char_idx += 1
        return translated_doc

    def translate_sentences(self, sentences: Union[str, List[str]], target_lang: str, source_lang: str = None,
                  show_progress_bar: bool = False, beam_size: int = 5, batch_size: int = 32, **kwargs):
        """
        This method translates individual sentences.

        :param sentences: A single sentence or a list of sentences to be translated
        :param source_lang: Source language for all sentences. If none, determines automatically the source language
        :param target_lang: Target language for the translation
        :param show_progress_bar: Show a progress bar
        :param beam_size: Size for beam search
        :param batch_size: Mini batch size
        :return: List of translated sentences
        """

        if source_lang == target_lang:
            return sentences

        is_single_sentence = False
        if isinstance(sentences, str):
            sentences = [sentences]
            is_single_sentence = True

        output = []
        if source_lang is None:
            #Determine languages for sentences
            src_langs = [self.language_detection(sent) for sent in sentences]
            logger.info("Detected languages: {}".format(set(src_langs)))

            #Group by languages
            lang2id = {}
            for idx, lng in enumerate(src_langs):
                if lng not in lang2id:
                    lang2id[lng] = []

                lang2id[lng].append(idx)

            #Translate language wise
            output = [None] * len(sentences)
            for lng, ids in lang2id.items():
                logger.info("Translate sentences of language: {}".format(lng))
                try:
                    grouped_sentences = [sentences[idx] for idx in ids]
                    translated = self.translate_sentences(grouped_sentences, source_lang=lng, target_lang=target_lang, show_progress_bar=show_progress_bar, beam_size=beam_size, batch_size=batch_size, **kwargs)
                    for idx, translated_sentences in zip(ids, translated):
                        output[idx] = translated_sentences
                except Exception as e:
                    logger.warning("Exception: "+str(e))
                    raise e
        else:
            #Sort by length to speed up processing
            length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            iterator = range(0, len(sentences_sorted), batch_size)
            if show_progress_bar:
                scale = min(batch_size, len(sentences))
                iterator = tqdm.tqdm(iterator, total=len(sentences)/scale, unit_scale=scale, smoothing=0)

            for start_idx in iterator:
                ### comment: step 2
                output.extend(self.translator.translate_sentences(sentences_sorted[start_idx:start_idx+batch_size], source_lang=source_lang, target_lang=target_lang, beam_size=beam_size, device=self.device, **kwargs))

            #Restore original sorting of sentences
            output = [output[idx] for idx in np.argsort(length_sorted_idx)]

        if is_single_sentence:
            output = output[0]

        return output



    def start_multi_process_pool(self, target_devices: List[str] = None):
        """
        Starts multi process to process the encoding with several, independent processes.
        This method is recommended if you want to encode on multiple GPUs. It is advised
        to start only one process per GPU. This method works together with encode_multi_process
        :param target_devices: PyTorch target devices, e.g. cuda:0, cuda:1... If None, all available CUDA devices will be used
        :return: Returns a dict with the target processes, an input queue and and output queue.
        """
        if target_devices is None:
            if torch.cuda.is_available():
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logger.info("CUDA is not available. Start 4 CPU worker")
                target_devices = ['cpu'] * 4

        logger.info("Start multi-process pool on devices: {}".format(', '.join(map(str, target_devices))))

        ctx = mp.get_context('spawn')
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for cuda_id in target_devices:
            p = ctx.Process(target=EasyNMT._encode_multi_process_worker, args=(cuda_id, self, input_queue, output_queue), daemon=True)
            p.start()
            processes.append(p)

        return {'input': input_queue, 'output': output_queue, 'processes': processes}

    def translate_multi_process(self, pool: Dict[str, object], documents: List[str], show_progress_bar: bool = True, chunk_size: int = None, **kwargs) -> List[str]:
        """
        This method allows to run encode() on multiple GPUs. The sentences are chunked into smaller packages
        and sent to individual processes, which encode these on the different GPUs. This method is only suitable
        for encoding large sets of sentences
        :param pool: A pool of workers started with SentenceTransformer.start_multi_process_pool
        :param chunk_size: Sentences are chunked and sent to the individual processes. If none, it determine a sensible size.
        """

        if chunk_size is None:
            chunk_size = min(math.ceil(len(documents) / len(pool["processes"]) / 10), 1000)

        logger.info("Chunk data into packages of size {}".format(chunk_size))

        input_queue = pool['input']
        last_chunk_id = 0

        for start_idx in range(0, len(documents), chunk_size):
            input_queue.put([last_chunk_id, documents[start_idx:start_idx+chunk_size], kwargs])
            last_chunk_id += 1

        output_queue = pool['output']
        results_list = sorted([output_queue.get() for _ in tqdm.tqdm(range(last_chunk_id), total=last_chunk_id, unit_scale=chunk_size, smoothing=0, disable=not show_progress_bar)], key=lambda chunk: chunk[0])
        translated = []
        for chunk in results_list:
            translated.extend(chunk[1])
        return translated

    def translate_stream(self, stream: Iterable[str], show_progress_bar: bool = True, chunk_size: int = 128, **kwargs) -> List[str]:
        batch = []
        for doc in tqdm.tqdm(stream, smoothing=0.0, disable=not show_progress_bar):
            batch.append(doc)

            if len(batch) >= chunk_size:
                translated = self.translate(batch, show_progress_bar=False, **kwargs)
                for trans_doc in translated:
                    yield trans_doc
                batch = []

        if len(batch) > 0:
            translated = self.translate(batch, show_progress_bar=False, **kwargs)
            for trans_doc in translated:
                yield trans_doc


    @staticmethod
    def stop_multi_process_pool(pool):
        """
        Stops all processes started with start_multi_process_pool
        """
        for p in pool['processes']:
            p.terminate()

        for p in pool['processes']:
            p.join()
            p.close()

        pool['input'].close()
        pool['output'].close()


    @staticmethod
    def _encode_multi_process_worker(target_device: str, model, input_queue, results_queue):
        """
        Internal working process to encode sentences in multi-process setup
        """
        model.device = target_device
        while True:
            try:
                id, documents, kwargs = input_queue.get()
                translated = model.translate(documents, **kwargs)
                results_queue.put([id, translated])
            except queue.Empty:
                break

    def language_detection(self, text: Union[str, List[str]]) -> str:
        """
       Given a text, detects the language code and returns the ISO language code.
       It test different language detectors, based on what is available:
       fastText, langid, langdetect.

       You can change the language detector order by changing model._lang_detectors
       :param text: Text or a List of Texts for which we want to determine the language
       :return: ISO language code
       """
        if isinstance(text, list):
            return [self.language_detection(doc) for doc in text]

        for lang_detector in self._lang_detectors:
            try:
                return lang_detector(text)
            except:
                pass

        raise Exception("No method for automatic language detection was found. Please install at least one of the following: fasttext (pip install fasttext), langid (pip install langid), or langdetect (pip install langdetect)")

    def language_detection_fasttext(self, text: str) -> str:
        """
        Given a text, detects the language code and returns the ISO language code. It supports 176 languages. Uses
        the fasttext model for language detection:
        https://fasttext.cc/blog/2017/10/02/blog-post.html
        https://fasttext.cc/docs/en/language-identification.html


        """
        if self._fasttext_lang_id is None:
            import fasttext
            fasttext.FastText.eprint = lambda x: None   #Silence useless warning: https://github.com/facebookresearch/fastText/issues/1067
            model_path = os.path.join(self._cache_folder, 'lid.176.ftz')
            if not os.path.exists(model_path):
                http_get('https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz', model_path)
            self._fasttext_lang_id = fasttext.load_model(model_path)

        return self._fasttext_lang_id.predict(text.lower().replace("\r\n", " ").replace("\n", " ").strip())[0][0].split('__')[-1]

    def language_detection_langid(self, text: str) -> str:
        import langid
        return langid.classify(text.lower().replace("\r\n", " ").replace("\n", " ").strip())[0]


    def language_detection_langdetect(self, text: str) -> str:
        import langdetect
        return langdetect.detect(text.lower().replace("\r\n", " ").replace("\n", " ").strip()).split("-")[0]


    def sentence_splitting(self, text: str, lang: str = None):
        if lang == 'th':
            from thai_segmenter import sentence_segment
            sentences = [str(sent) for sent in sentence_segment(text)]
        elif lang in ['ar', 'jp', 'ko', 'zh']:
            sentences = list(re.findall(u'[^!?。\.]+[!?。\.]*', text, flags=re.U))
        else:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

            sentences = nltk.sent_tokenize(text)

        return sentences


    @property
    def lang_pairs(self) -> FrozenSet[str]:
        """
        Returns all allowed languages directions for the loaded model
        """
        return self._lang_pairs

    def get_languages(self, source_lang: str = None, target_lang: str = None) -> List[str]:
        """
        Returns all available languages supported by the model
        :param source_lang:  If not None, then returns all languages to which we can translate for the given source_lang
        :param target_lang:  If not None, then returns all languages from which we can translate for the given target_lang
        :return: Sorted list with the determined languages
        """

        langs = set()
        for lang_pair in self.lang_pairs:
            source, target = lang_pair.split("-")

            if source_lang is None and target_lang is None:
                langs.add(source)
                langs.add(target)
            elif target_lang is not None and target == target_lang:
                langs.add(source)
            elif source_lang is not None and source == source_lang:
                langs.add(target)

        return sorted(list(langs))


    def save(self, output_path):
        os.makedirs(output_path, exist_ok=True)

        filepath = os.path.join(output_path, 'easynmt.json')

        config = {
            'model_class': fullname(self.translator),
            'lang_pairs': list(self.lang_pairs),
            'model_args': self.translator.save(output_path)
        }

        with open(filepath, 'w') as fOut:
            json.dump(config, fOut)
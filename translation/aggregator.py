"""
Module name: translator

This module contains the Translator class, which is used for managing translation using the NLLB model.

Classes:
    Translator: This class is used for managing translation using the NLLB model.
    - __init__(self, available_langs:list, model:str="facebook/nllb-200-distilled-600M"): This is the constructor for the Translator class. It takes a required parameter `available_langs`, which is a list of languages that can be used for translation, and an optional parameter `model`, which specifies the NLLB model to be used for translation. The default value is "facebook/nllb-200-distilled-600M".
    - get_source_language(self, text_input, k=1): This method detects the language of the input text and returns the result.
    - get_toxicity(self, text_input): This method detects the language of the input text, calculates the toxicity of the input text, and returns the results.
    - get_translation(self, text_input, langs_out:list=None, max_length:int=30): This method translates the input text to the languages specified in `langs_out`.

"""
import numpy as np

from translation.nllb import NLLB
from translation.lid import LanguageDetection
from translation.toxicity import ToxicityCounter
from utils.logger import setup_custom_logger


class Aggregator:
    def __init__(
        self, available_langs: list, model: str = "facebook/nllb-200-distilled-600M"
    ):
        """
        Initializes the Translator object.

        Parameters:
            available_langs (list): A list of languages that can be used for translation.
            model (str): This is an optional parameter that specifies the NLLB model to be used for translation. The default value is "facebook/nllb-200-distilled-600M".
        """
        # configuration
        self.available_langs = available_langs
        self.model = NLLB(available_langs, model)
        self.lid = LanguageDetection()
        self.tox = ToxicityCounter()

        # logger
        self.logger = setup_custom_logger("Translator")
        self.logger.info(
            "Available langs to translate: {}".format(self.available_langs)
        )

    def get_source_language(self, text_input, k=1):
        """
        Detects the language of the input text and returns the result.

        Parameters:
            text_input (type): The input text whose language is to be detected.
            k (int): This is an optional parameter that specifies the number of languages to be detected. The default value is 1.

        Returns:
            dict: The language detection and input text.
        """
        # convert string to list of strings
        input_text = [text_input] if type(text_input) == str else text_input

        # detect input language
        lang_in_detection = self.lid.detect(input_text)

        # prepare response dict data
        source_language = {"text": input_text, "lid": lang_in_detection["lid"]}

        return source_language

    def get_toxicity(self, text_input):
        """
        Detects the language of the input text, calculates the toxicity of the input text, and returns the results.

        Parameters:
            text_input (type): The input text whose toxicity is to be calculated.

        Returns:
            dict: The language detection, input text, and toxicity calculation results.
        """
        # convert string to list of strings
        input_text = [text_input] if type(text_input) == str else text_input

        # detect input language
        lang_in_detection = self.lid.detect(input_text)

        # prepare response dict data
        toxicity = {
            "text": input_text,
            "lid": lang_in_detection["lid"],
            "toxicity": self.tox.etox_single_list(
                input_text, lang_in_detection["lid"]["lang"]
            )["toxicity"],
        }

        return toxicity

    def get_translation(self, text_input, langs_out: list = None, max_length: int = 30):
        """
        Translates the input text to the languages specified in `langs_out`.

        Parameters:
            text_input (type): The input text to be translated.
            langs_out (list): This is an optional parameter that specifies the languages to which the input text is to be translated. If not specified, the input text will be translated to all available languages.
            max_length (int): This is an optional parameter that specifies the maximum length of the translated text. The default value is 30.

        Returns:
            dict: The input text, language detection results, and translations.
        """
        # convert string to list of strings
        input_text = [text_input] if type(text_input) == str else text_input

        # translate to these languages
        translation_langs = self.available_langs if langs_out is None else langs_out

        # prepare response dict data
        translations = {}
        translations["input"] = {"text": input_text, "lang_out": translation_langs}

        # detect input language
        lang_in_detection = self.lid.detect(input_text)
        translations["input"]["lid"] = lang_in_detection["lid"]

        # get translations in translation_langs
        translations["result"] = {}
        for lang_out in translation_langs:
            translations["result"][lang_out] = {
                "text": self.model.translate(input_text, lang_out, lang_out, max_length)
            }

        # modify translated text from lang_in
        for i in range(len(translations["input"]["text"])):
            lang_detected = translations["input"]["lid"]["lang"][i]
            if lang_detected in translation_langs:
                translations["result"][lang_detected]["text"][i] = translations[
                    "input"
                ]["text"][i]

        # get toxicitys
        translations["input"]["toxicity"] = self.tox.etox_single_list(
            input_text, translations["input"]["lid"]["lang"]
        )["toxicity"]
        input_tox_count = np.array(translations["input"]["toxicity"]["count"])
        for lang in translations["result"].keys():
            translated_text = translations["result"][lang]["text"]
            translations["result"][lang]["toxicity"] = self.tox.etox_single(
                translated_text, lang
            )["toxicity"]

            diff = (
                np.array(translations["result"][lang]["toxicity"]["count"])
                - input_tox_count
            )
            translations["result"][lang]["toxicity"]["added_toxicity"] = [
                bool(b) for b in list(diff > 0)
            ]
        return translations

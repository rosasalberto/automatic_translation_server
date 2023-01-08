"""
Module name: lang_tokenizers

This module contains the LangTokenizers class, which is used for managing tokenizers for different languages.

Classes:
    LangTokenizers: This class is used for managing tokenizers for different languages.
    - __init__(self, available_langs:list, model:str="facebook/nllb-200-distilled-600M"): This is the constructor for the LangTokenizers class. It takes a required parameter `available_langs`, which is a list of languages for which tokenizers are to be initialized, and an optional parameter `model`, which specifies the NLLB model to be used for creating the tokenizers. The default value is "facebook/nllb-200-distilled-600M".
    - _initialize_tokenizers(self, model:str): This method initializes the tokenizers for the languages specified in the `available_langs` parameter of the constructor.
    - get_tokenizer(self, lang:str): This method returns the tokenizer for the specified language.
    - decode(self, tokenizer, translated_tokens): This method decodes the translated tokens using the specified tokenizer.

"""
from transformers import NllbTokenizerFast

from utils.langs import lang_list


class LangTokenizers:
    def __init__(
        self, available_langs: list, model: str = "facebook/nllb-200-distilled-600M"
    ):
        """
        Initializes the LangTokenizers object.

        Parameters:
            available_langs (list): A list of languages for which tokenizers are to be initialized.
            model (str): This is an optional parameter that specifies the NLLB model to be used for creating the tokenizers. The default value is "facebook/nllb-200-distilled-600M".
        """
        self.available_langs = available_langs
        self._initialize_tokenizers(model)

    def _initialize_tokenizers(self, model: str):
        """
        Initializes the tokenizers for the languages specified in the `available_langs` parameter of the constructor.

        Parameters:
            model (str): The NLLB model to be used for creating the tokenizers.
        """
        self.tokenizer = {}
        for lang in self.available_langs:
            assert lang in lang_list, "INVALID_LANG: {} not in langs list".format(lang)
            self.tokenizer[lang] = NllbTokenizerFast.from_pretrained(model, use_auth_token=False, src_lang=lang)

    def get_tokenizer(self, lang: str):
        """
        Returns the tokenizer for the specified language.

        Parameters:
            lang (str): The language for which the tokenizer is to be returned.

        Returns:
            type: The tokenizer for the specified language.
        """
        assert (
            lang in self.tokenizer.keys()
        ), "INVALID_LANG: {} not in tokenizers list".format(lang)
        return self.tokenizer[lang]

    def decode(self, tokenizer, translated_tokens):
        """
        Decodes the translated tokens using the specified tokenizer.

        Parameters:
            tokenizer (type): The tokenizer to be used for decoding the translated tokens.
            translated_tokens (type): The translated tokens to be decoded.

        Returns:
            type: The decoded text.
        """
        tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

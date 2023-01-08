"""
Module name: nllb

This module contains the NLLB class, which is used for translation using the NLLB model.

Classes:
    NLLB: This class is used for translation using the NLLB model.
    - __init__(self, model:str="facebook/nllb-200-distilled-600M"): This is the constructor for the NLLB class. It takes an optional parameter `model`, which specifies the NLLB model to be used for translation. The default value is "facebook/nllb-200-distilled-600M".
    - generate(self, inputs, lang_out_id, max_length:int=30): This method translates the input text using the NLLB model.
    - translate(self, input_text, lang_in:str, lang_out:str, max_length=30): This method translates the input text from the language specified in `lang_in` to the language specified in `lang_out`.


"""
from transformers import AutoModelForSeq2SeqLM

from translation.tokenizers import LangTokenizers
from config import translation_langs


class NLLB:
    def __init__(self, available_langs, model: str = "facebook/nllb-200-distilled-600M"):
        """
        Initializes the NLLB object.

        Parameters:
            model (str): This is an optional parameter that specifies the NLLB model to be used for translation. The default value is "facebook/nllb-200-distilled-600M".
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, use_auth_token=False)
        self.tokenizers = LangTokenizers(available_langs)

    def generate(self, inputs, lang_out_id, max_length: int = 30):
        """
        Translates the input text using the NLLB model.

        Parameters:
            inputs (type): The input text to be translated.
            lang_out_id (type): The ID of the output language.
            max_length (int): An optional parameter that specifies the maximum length of the output text. The default value is 30.

        Returns:
            type: The translated text.
        """
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=lang_out_id, max_length=max_length
        )
        return translated_tokens

    def translate(self, input_text, lang_in: str, lang_out: str, max_length=30):
        """
        Translates the input text from the language specified in `lang_in` to the language specified in `lang_out`.

                Parameters:
            input_text (type): The input text to be translated.
            lang_in (str): The language of the input text.
            lang_out (str): The language to which the input text is to be translated.
            max_length (int): This is an optional parameter that specifies the maximum length of the translated text. The default value is 30.

        Returns:
            type: The translated text.
        """
        # select tokenizer
        selected_tokenizer = self.tokenizers.get_tokenizer(lang_in)

        # tokenize input
        inputs = selected_tokenizer(input_text, return_tensors="pt", padding=True)

        # get id lang out
        lang_out_id = selected_tokenizer.lang_code_to_id[lang_out]

        # translate tokens
        translated_tokens = self.generate(
            inputs, lang_out_id, max_length=max_length
        )

        # decoded translated tokens
        decoded_translation = selected_tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )

        return decoded_translation
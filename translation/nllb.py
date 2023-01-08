"""
Module name: nllb

This module contains the NLLB class, which is used for translation using the NLLB model.

Classes:
    NLLB: This class is used for translation using the NLLB model.
    - __init__(self, model:str="facebook/nllb-200-distilled-600M"): This is the constructor for the NLLB class. It takes an optional parameter `model`, which specifies the NLLB model to be used for translation. The default value is "facebook/nllb-200-distilled-600M".
    - translate(self, inputs, lang_out_id, max_length:int=30): This method translates the input text using the NLLB model.

"""
from transformers import AutoModelForSeq2SeqLM


class NLLB:
    def __init__(self, model: str = "facebook/nllb-200-distilled-600M"):
        """
        Initializes the NLLB object.

        Parameters:
            model (str): This is an optional parameter that specifies the NLLB model to be used for translation. The default value is "facebook/nllb-200-distilled-600M".
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model, use_auth_token=False)

    def translate(self, inputs, lang_out_id, max_length: int = 30):
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

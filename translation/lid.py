"""
Module name: language_detection

This module contains the LanguageDetection class, which is used for detecting the language of input text.

Classes:
    LanguageDetection: This class is used to detect the language of input text.
    - __init__(self, model:str="meta-lid"): This is the constructor for the LanguageDetection class. It takes an optional parameter `model`, which specifies the model to be used for language detection. The default value is "meta-lid".
    - format(self, input_text:list, result:tuple): This method formats the result of language detection into a dictionary with the input text and the detected language and scores.
    - detect(self, input_text:list, k:int=1): This method takes a list of input text and an optional parameter `k`, which specifies the number of languages to be detected. It returns a dictionary with the input text and the detected language and scores.

"""
import fasttext

from config import lid_path


class LanguageDetection:
    def __init__(self, model: str = "meta-lid"):
        """
        Initializes the LanguageDetection object.

        Parameters:
            model (str): This is an optional parameter that specifies the model to be used for language detection. The default value is "meta-lid".
        """
        if model == "meta-lid":
            self.model = fasttext.load_model(lid_path)
        else:
            raise ValueError("Model not supported")

    def format(self, input_text: list, result: tuple):
        """
        Formats the result of language detection into a dictionary with the input text and the detected language and scores.

        Parameters:
            input_text (list): A list of input text.
            result (tuple): A tuple containing the detected language and scores.

        Returns:
            dict: A dictionary with the input text and the detected language and scores.
        """
        d = {
            "input_text": input_text,
            "lid": {
                "lang": [lang[0].replace("__label__", "") for lang in result[0]],
                "scores": [float(score[0]) for score in result[1]],
            },
        }
        return d

    # input_text has to be a list of strings
    def detect(self, input_text: list, k: int = 1):
        """
        Detects the language of input text.

        Parameters:
            input_text (list): A list of input text.
            k (int): An optional parameter that specifies the number of languages to be detected. The default value is 1.

        Returns:
            dict: A dictionary with the input text and the detected language and scores.
        """
        lang_detection = self.model.predict(input_text, k)
        return self.format(input_text, lang_detection)

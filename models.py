"""
Module name: models

This module contains the models used in the translation service.

Classes:
    ToxicityInput: This class represents the input to the toxicity prediction service.
    ToxicityResult: This class represents the result of the toxicity prediction service.
    LID: This class represents the language detection result.
    Input: This class represents the input to the translation service.
    Result: This class represents the result of the translation service.
    Language: This class represents the result of the translation service for all languages.
    Translation: This class represents the input and result of the translation service.
    Toxicity: This class represents the input to the toxicity prediction service.
    LIDInput: This class represents the input to the language detection service.

"""

from fastapi import Body
from typing import List, Union
from pydantic import BaseModel
from config import translation_langs


class ToxicityInput(BaseModel):
    """
    This class represents the input to the toxicity prediction service.

    Attributes:
        words (List[List[str]]): The list of words to be checked for toxicity.
        count (List[int]): The list of counts of each word in the `words` attribute.
    """

    words: List[List[str]]
    count: List[int]


class ToxicityResult(BaseModel):
    """
    This class represents the result of the toxicity prediction service.

    Attributes:
        words (List[List[str]]): The list of words that were checked for toxicity.
        count (List[int]): The list of counts of each word in the `words` attribute.
        added_toxicity (List[bool]): The list of booleans indicating whether each word in the `words` attribute is toxic.
    """

    words: List[List[str]]
    count: List[int]
    added_toxicity: List[bool]


class LID(BaseModel):
    """
    This class represents the language detection result.

    Attributes:
        lang (List[str]): The list of languages detected.
        scores (List[float]): The list of scores indicating the confidence of the language detection.
    """

    lang: List[str]
    scores: List[float]


class Input(BaseModel):
    """
    This class represents the input to the translation service.

    Attributes:
        text (List[str]): The list of texts to be translated.
        lang_out (List[str]): The list of languages to translate the texts to.
        lid (LID): The language detection result.
        toxicity (ToxicityInput): The input to the toxicity prediction service.
    """

    text: List[str]
    lang_out: List[str]
    lid: LID
    toxicity: ToxicityInput


class Result(BaseModel):
    """
    This class represents the result of the translation service.

    Attributes:
        text (List[str]): The list of translated texts.
        toxicity (ToxicityResult): The result of the toxicity prediction service.
    """

    text: List[str]
    toxicity: ToxicityResult


langs_model_str = ""
for lang in translation_langs:
    langs_model_str += f"{lang}: Union[Result, None] = None\n"


class Language(BaseModel):
    """
    This class represents the result of the translation service for all languages.

    Attributes:
        lang (Union[Result, None]): The result of the translation service for the language specified by `lang`.
    """

    exec(langs_model_str)


class Translation(BaseModel):
    """
    This class represents the input and result of the translation service.

    Attributes:
        input (Input): The input to the translation service.
        result (Language): The result of the translation service for all languages.
    """

    input: Input
    result: Language


class Toxicity(BaseModel):
    """
    This class represents the input to the toxicity prediction service.

    Attributes:
        text (List[str]): The list of texts to be checked for toxicity.
        lid (LID): The language detection result.
        toxicity (ToxicityInput): The input to the toxicity prediction service.
    """

    text: List[str]
    lid: LID
    toxicity: ToxicityInput


class LIDInput(BaseModel):
    """
    This class represents the input to the language detection service.

    Attributes:
        text (List[str]): The list of texts to be language detected.
        lid (LID): The language detection result.
    """

    text: List[str]
    lid: LID


class RequestBodyInput:
    """
    Query parameters for the `toxicity` and `lid` endpoint.

    Args:
        input_text (str): The input text.
    """

    def __init__(
        self,
        input_text: str = Body(
            "Attention is all you need. Vaya mierda de dia que hace",
            description="Text separated by '.' to translate",
        ),
    ):
        self.input_text = input_text


class RequestBodyTranslate:
    """
    Query parameters for the `translate` endpoint.

    Args:
        input_text (str): The input text.
        langs_out (str): The languages to translate to, separated by ',' using FLORES-200 code. Type 'all' for translating to all languages.
    """

    def __init__(
        self,
        input_text: str = Body(
            "Attention is all you need. Vaya mierda de dia que hace",
            description="Text separated by '.' to translate",
        ),
        langs_out: str = Body(
            "spa_Latn, eng_Latn",
            description="Languages to translate separated by ',' using FLORES-200 code. Type 'all' for translating to all languages",
        ),
    ):
        self.input_text = input_text
        self.langs_out = langs_out

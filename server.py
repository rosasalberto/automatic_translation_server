from fastapi import FastAPI, Depends

from translation.aggregator import Aggregator
from config import translation_langs
from models import (
    Translation,
    Toxicity,
    LIDInput,
    RequestBodyInput,
    RequestBodyTranslate,
)
from utils.formatter import get_input_text_formatted

tags_metadata = [
    {
        "name": "translate",
        "description": "Translate text to 200 languages automatically without specifying the source language.",
    },
    {
        "name": "toxicity",
        "description": "Get toxicity in texts without specifying the source language.",
    },
    {
        "name": "language detection",
        "description": "Detect source language of a text from 200 languages.",
    },
]

app = FastAPI(
    title="Automatic Translations tools",
    description="Automatic translation tools, translate text to 200 languages, detect source language, get toxicity from text:",
    version="0.1.0",
    openapi_tags=tags_metadata,
    contact={
        "name": "Alberto Rosas, Co-CEO Gamium",
        "email": "alberto.rosas@gamiumworld.com",
    },
)


@app.exception_handler(Exception)
async def handle_exception(request, exc):
    """
    This function handles exceptions raised in the endpoint functions.

    Args:
        request (Request): The request object.
        exc (Exception): The exception object.

    Returns:
        Dict: The response to be sent to the client.
    """
    return {"error": str(exc)}


### Langs endpoint ###


@app.get(
    "/langs",
    tags=["langs"],
    summary="Get available languages",
    description="Get all the availaible translation languages",
    response_description="Returns a list with all the available languages using FLORES-200 code",
)
def langs():
    """
    This function returns the list of available languages for translation.

    Returns:
        List[str]: The list of available languages using FLORES-200 code.
    """
    return translation_langs


### Translation endpoint ###


@app.post(
    "/translate",
    tags=["translate"],
    summary="Translate text",
    description="Automatic translation using NLLB model from Meta AI. Translate input_text in langs_out languages. Available languages: {}".format(
        translation_langs
    ),
    response_description="Returns translated text in all the languages specified",
    response_model=Translation,
)
def translate(params: RequestBodyTranslate = Depends()):
    """
    This function translates the input text to the specified languages.

    Args:
        params (TranslateQueryParams): The query parameters for the `translate` function.

    Returns:
        Translation: The result of the translation service.

    Raises:
        Exception: Any exception raised during the translation process.
    """
    input_text_ = get_input_text_formatted(params.input_text)
    langs_out_ = (
        None
        if (params.langs_out == "all" or params.langs_out is None)
        else params.langs_out.replace(" ", "").split(",")
    )

    translation = aggregator.get_translation(input_text_, langs_out_)
    return translation


### Toxicity endpoint ###


@app.post(
    "/toxicity",
    tags=["toxicity"],
    summary="Detect toxicity in text",
    description="Get toxicity in texts without specifying the source language",
    response_description="Returns toxic words in input_text",
    response_model=Toxicity,
)
def toxicity(params: RequestBodyInput = Depends()):
    """
    This function checks the input text for toxicity.

    Args:
        params (InputTextBody): The query parameters for the `toxicity` function.

    Returns:
        Toxicity: The result of the toxicity detection service.
    """
    input_text_ = get_input_text_formatted(params.input_text)

    toxicity = aggregator.get_toxicity(input_text_)
    return toxicity


### LID endpoint ###


@app.post(
    "/lid",
    tags=["language detection"],
    summary="Detect source language of a text from 200 languages",
    description="Detect source language of a text from 200 languages (FLORES-200)",
    response_description="Returns language detected and confidence score",
    response_model=LIDInput,
)
def lid(params: RequestBodyInput = Depends()):
    """
    This function detects the source language of the input text.

    Args:
        params (InputTextBody): The query parameters for the `lid` function.

    Returns:
        LIDInput: The result of the language detection service.
    """
    input_text_ = get_input_text_formatted(params.input_text)

    lid = aggregator.get_source_language(input_text_)
    return lid


aggregator = Aggregator(translation_langs)

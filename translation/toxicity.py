"""
Module name: toxicity_counter

This module contains the ToxicityCounter class, which is used for counting the number of toxic words in a given input text.

Classes:
    ToxicityCounter: This class is used for counting the number of toxic words in a given input text.
    - __init__(self): This is the constructor for the ToxicityCounter class. It initializes the class and loads the toxicity word lists for different languages.
    - configure(self): This method is used to configure the ToxicityCounter object. It loads the toxicity word lists for different languages.
    - txt_format(self, input_text:list): This method formats the input text by lowercasing it, replacing punctuation with spaces, and adding space padding.
    - import_toxicity_list_file(self, toxicity_filename, verbose=False): This method loads the toxicity word list for the specified language from a raw text file.
    - token_checker(self, string, toxic_word_list): This method checks the input string for the presence of toxic words from the specified toxic word list.
    - etox_single(self, input_data:list, input_lang:str): This method counts the number of toxic words in a single input text.
    - etox_single_list(self, input_data:list, input_langs:list): This method counts the number of toxic words in a list of input texts.

"""
import pandas as pd

from config import toxicity_files


class ToxicityCounter:
    def __init__(self):
        """
        Initializes the ToxicityCounter object and loads the toxicity word lists for different languages.
        """
        self.configure()

    def configure(self):
        """
        Configures the ToxicityCounter object by loading the toxicity word lists for different languages.
        """
        self.toxic_words_by_langs = {}
        for lang in toxicity_files.keys():
            self.toxic_words_by_langs[lang] = self.import_toxicity_list_file(
                toxicity_files[lang]
            )

    def txt_format(self, input_text: list):
        """
        Formats the input text by lowercasing it, replacing punctuation with spaces, and adding space padding.

        Parameters:
            input_text (list): The input text to be formatted.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the formatted input text.
        """
        df_gen = pd.DataFrame({"string_raw": input_text})
        df_gen.loc[:, "string_raw"] = df_gen["string_raw"].str.lower()
        df_gen.loc[:, "string_raw"] = df_gen["string_raw"].str.replace(
            "[\W+]", " ", regex=True
        )
        df_gen.loc[:, "string_raw"] = df_gen["string_raw"].apply(
            lambda x: " " + str(x) + " "
        )
        return df_gen

    def import_toxicity_list_file(self, toxicity_filename, verbose=False):
        """
        Loads the toxicity word list for the specified language from a raw text file.

         Parameters:
            toxicity_filename (str): The file containing the toxicity word list for the specified language.
            verbose (bool): This is an optional parameter that specifies whether to print the toxicity word list after it is loaded. The default value is False.

        Returns:
            list: A list containing the toxicity words for the specified language.
        """
        filename = toxicity_filename
        with open(filename, encoding="utf-8") as f:
            data = f.readlines()
            t = []
            for line in data:
                t += [line.rstrip("\n")]
            f.close()
        if verbose == True:
            print(f"{toxicity_filename} TOXIC WORD LIST length = {len(t)} \n", t)

        deduplicated_list = list(set(t))
        return deduplicated_list

    def token_checker(self, string, toxic_word_list):
        """
        Checks the input string for the presence of toxic words from the specified toxic word list.

        Parameters:
            string (str): The input string to be checked for toxic words.
            toxic_word_list (list): The toxic word list to be used for checking the input string.

        Returns:
            list: A list containing the toxic words found in the input string.
        """
        l = []
        toxic_word_list = [" " + x.lower() + " " for x in toxic_word_list]
        string = " " + string.lower() + " "
        for w in toxic_word_list:
            if string.__contains__(w):
                l += [w]
        l = [
            x.strip(" ") for x in l
        ]  # removes the token-marking " " pads from the toxic words afterwards
        return l

    def etox_single(self, input_data: list, input_lang: str):
        """
        Counts the number of toxic words in a single input text.

        Parameters:
            input_data (list): The input text to be checked for toxic words.
            input_lang (str): The language of the input text.

        Returns:
            dict: A dictionary containing the toxic words and their count in the input text.
        """
        # clean up the strings before toxicity check:
        # lowercases everying, replacing punctuation to spaces,
        df_eval = self.txt_format(input_data)

        # Load the toxicity word list:
        toxicity_list = self.toxic_words_by_langs[input_lang]

        # uses a different tokenizer depending on input parameter
        df_eval.loc[:, ["matched_toxicity_list"]] = df_eval["string_raw"].apply(
            self.token_checker, toxic_word_list=toxicity_list
        )

        # get important dict values
        toxic_words = df_eval["matched_toxicity_list"].values
        toxic_words = [t for t in toxic_words]
        toxic_words_count = [len(t) for t in toxic_words]

        toxicity = {"toxicity": {"words": toxic_words, "count": toxic_words_count}}
        return toxicity

    def etox_single_list(self, input_data: list, input_langs: list):
        """
        Counts the number of toxic words in a list of input texts.

        Parameters:
            input_data (list): The list of input texts to be checked for toxic words.
            input_langs (list): The languages of the input texts.

        Returns:
            dict: A dictionary containing the toxic words and their count in the input texts.
        """
        words = []
        counts = []
        for i in range(len(input_data)):
            d = self.etox_single([input_data[i]], input_langs[i])
            words.append(d["toxicity"]["words"][0])
            counts.append(d["toxicity"]["count"][0])

        # get important dict values
        toxicity = {"toxicity": {"words": words, "count": counts}}
        return toxicity

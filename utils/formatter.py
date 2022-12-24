def get_input_text_formatted(input_text):
    input_text_ = input_text.replace(". ", ".").split(".")
    input_text_ = [inp for inp in input_text_ if inp != ""]
    return input_text_

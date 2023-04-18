import json
import os
import augly.text as textaugs


class TextProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

    def process_text(self, text_file):
        # Read in text file
        with open(os.path.join(self.input_path, text_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Processing of captions
        processed_captions = []
        for item in data:
            caption = item["caption"]
            processed_caption = self._process(caption)
            item["caption"] = processed_caption
            processed_captions.append(item)

        # Save processed captions to file
        output_file_path = os.path.join(self.output_path, text_file)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_captions, f, indent=4,
                      separators=(',', ': '), ensure_ascii=False)
            f.write('\n')

    def _process(self, text):
        # Add text processing methods here
        # Return the processed text
        return text.upper()  # Example: Convert all text to uppercase


class ChangeCase(TextProcessor):
    def __init__(self, input_path, output_path,
                 granularity: str = "char",
                 cadence: float = 1.0,
                 case: str = "random",
                 p: float = 1.0
                 ):
        """ Change case for the text

        Args:
            input_path (_type_): Path of input file
            output_path (_type_): Path of output file
            granularity (str, optional): "char"(case of random chars is changed), 
                                        "word"(case of random words is changed), 
                                        "all"(case of the entire text is changed). 
                                        Defaults to "char".
            cadence (float, optional): How frequent (i.e. between this many characters/words) to change the case. 
                                    Must be at least 1.0.
                                    Non-integer values are used as an 'average' cadence. 
                                    Not used for granularity 'all'.
                                    Defaults to 1.0.
            case (str, optional): The case to change words to; 
                                valid values are 'lower', 'upper', 'title', or 'random' 
                                Defaults to "random".
            p (float, optional): The probability of the transform being applied.
                                Defaults to 1.0.
        """
        super().__init__(input_path, os.path.join(output_path, "ChangeCase"))
        self.aug = textaugs.ChangeCase(
            granularity=granularity, cadence=cadence, case=case, p=p)

    def _process(self, text):
        # Change case for the text
        return self.aug(text)

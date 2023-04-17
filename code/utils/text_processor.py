import os

class TextProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)

    def process_text(self, text_file):
        # Read in text file
        with open(os.path.join(self.input_path, text_file), "r") as f:
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
        with open(output_file_path, "w") as f:
            json.dump(processed_captions, f)
    
    def _process(self, text):
        # Add text processing methods here
        # Return the processed text
        return text.upper()  # Example: Convert all text to uppercase

class AddNoise(TextProcessor):
    def __init__(self, input_path, output_path, noise_type):
        super().__init__(input_path, os.path.join(output_path, "AddNoise"))
        self.noise_type = noise_type

    def _process(self, text):
        # Add noise to text
        if self.noise_type == 'random':
            processed_text = add_random_noise(text)
        elif self.noise_type == 'spelling':
            processed_text = add_spelling_noise(text)
        else:
            raise ValueError("Invalid noise type")
        return processed_text

class Replace(TextProcessor):
    def __init__(self, input_path, output_path, replace_type):
        super().__init__(input_path, os.path.join(output_path, "Replace"))
        self.replace_type = replace_type
        
    def _process(self, text):
        # Replace words in text
        if self.replace_type == 'synonyms':
            processed_text = replace_with_synonyms(text)
        elif self.replace_type == 'antonyms':
            processed_text = replace_with_antonyms(text)
        else:
            raise ValueError("Invalid replace type")
        return processed_text

class Translate(TextProcessor):
    def __init__(self, input_path, output_path, src_lang, tgt_lang):
        super().__init__(input_path, os.path.join(output_path, "Translate"))
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def _process(self, text):
        # Translate text from source language to target language
        processed_text = translate_text(text, self.src_lang, self.tgt_lang)
        return processed_text
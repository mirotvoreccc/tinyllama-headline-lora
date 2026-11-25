import unittest

from src.pipeline import encode_example


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "PROMPT"

    def encode(self, text, add_special_tokens):
        return [ord(char) for char in text]


class EncodingTests(unittest.TestCase):
    def test_only_answer_tokens_have_labels(self):
        encoded = encode_example(FakeTokenizer(), "article", "Hi", max_length=16)
        self.assertEqual(len(encoded["input_ids"]), len(encoded["labels"]))
        self.assertEqual(encoded["labels"][-3:], [ord("H"), ord("i"), 99])
        self.assertTrue(all(label == -100 for label in encoded["labels"][:-3]))


if __name__ == "__main__":
    unittest.main()

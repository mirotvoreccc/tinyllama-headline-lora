import unittest

from datasets import Dataset

from src.evaluate_existing import select_held_out
from src.pipeline import unique_indices


class HeldOutTests(unittest.TestCase):
    def test_training_and_validation_skip_text_duplicates(self):
        rows = Dataset.from_dict({"text": ["First", "first  ", "Second", "Third", "second", "Fourth"]})
        seen = set()
        train_indices = unique_indices(rows, 2, seen)
        validation_indices = unique_indices(rows, 2, seen, start=train_indices[-1] + 1)
        self.assertEqual(train_indices, [0, 2])
        self.assertEqual(validation_indices, [3, 5])

    def test_excludes_training_and_previously_viewed_text(self):
        train = Dataset.from_dict({"text": ["Training article"], "title": ["A"]})
        texts = [f"Article {i}" for i in range(305)]
        texts[300] = " training   ARTICLE "
        texts[301] = "article 10"
        texts[302] = "New story"
        texts[303] = "new  STORY"
        texts[304] = "Another new story"
        test = Dataset.from_dict({"text": texts, "title": ["B"] * len(texts)})
        self.assertEqual(select_held_out(train, test, count=2, start=300), [302, 304])

    def test_new_test_slice_starts_after_preliminary_inspection(self):
        rows = Dataset.from_dict({"text": [f"Article {i}" for i in range(503)]})
        self.assertEqual(unique_indices(rows, 2, set(), start=500), [500, 501])


if __name__ == "__main__":
    unittest.main()

import unittest

from knowledge_brain.qa import answer_question, best_chunks, split_text


class QATests(unittest.TestCase):
    def test_split_text_returns_overlapping_chunks(self):
        text = " ".join(f"word{i}" for i in range(30))
        chunks = split_text(text, chunk_size=10, overlap=2)
        self.assertGreater(len(chunks), 1)
        self.assertIn("word8 word9", chunks[1])

    def test_best_chunks_prefers_relevant_content(self):
        chunks = [
            "The quarterly sales plan focuses on LATAM growth and enterprise deals.",
            "We discussed team lunch and office events for Friday.",
        ]
        best = best_chunks("What is the sales plan for enterprise growth?", chunks, top_k=1)
        self.assertEqual(best[0], chunks[0])

    def test_answer_question_handles_missing_matches(self):
        answer = answer_question("Tell me about quantum teleportation", ["Budget summary and hiring plan"])
        self.assertIn("could not find", answer)


if __name__ == "__main__":
    unittest.main()

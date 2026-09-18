import unittest

from knowledge_brain.qa import (
    answer_question,
    best_chunks,
    chunks_from_uploaded_files,
    split_text,
    upload_signature,
)


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

    def test_split_text_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("hello world", chunk_size=5, overlap=5)

    def test_chunks_from_uploaded_files_returns_empty_without_uploads(self):
        self.assertEqual(chunks_from_uploaded_files([]), [])

    def test_upload_signature_is_empty_without_uploads(self):
        self.assertEqual(upload_signature([]), ())

    def test_upload_signature_changes_when_content_changes(self):
        class UploadedFileStub:
            def __init__(self, name, content):
                self.name = name
                self._content = content

            def getvalue(self):
                return self._content

        same_name_a = UploadedFileStub("slides.pdf", b"version-a")
        same_name_b = UploadedFileStub("slides.pdf", b"version-b")
        self.assertNotEqual(upload_signature([same_name_a]), upload_signature([same_name_b]))

    def test_upload_signature_is_order_insensitive(self):
        class UploadedFileStub:
            def __init__(self, name, content):
                self.name = name
                self._content = content

            def getvalue(self):
                return self._content

        file_a = UploadedFileStub("a.pdf", b"a")
        file_b = UploadedFileStub("b.pdf", b"b")
        self.assertEqual(upload_signature([file_a, file_b]), upload_signature([file_b, file_a]))


if __name__ == "__main__":
    unittest.main()

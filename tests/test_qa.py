import unittest
from io import BytesIO

from knowledge_brain.qa import (
    answer_question,
    best_chunks,
    chunks_from_uploaded_files,
    split_text,
    upload_signature,
)


class UploadedFileStub:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


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

    def test_answer_question_returns_joined_matches(self):
        answer = answer_question(
            "What are the enterprise growth priorities?",
            [
                "Enterprise growth priorities include account expansion.",
                "Another enterprise growth focus is partner channels.",
            ],
        )
        self.assertIn("\n\n", answer)
        self.assertIn("Enterprise growth priorities", answer)

    def test_split_text_rejects_invalid_overlap(self):
        with self.assertRaises(ValueError):
            split_text("hello world", chunk_size=5, overlap=5)

    def test_chunks_from_uploaded_files_returns_empty_without_uploads(self):
        self.assertEqual(chunks_from_uploaded_files([]), [])

    def test_upload_signature_is_empty_without_uploads(self):
        self.assertEqual(upload_signature([]), ())

    def test_upload_signature_changes_when_content_changes(self):
        same_name_a = UploadedFileStub("slides.pdf", b"version-a")
        same_name_b = UploadedFileStub("slides.pdf", b"version-b")
        self.assertNotEqual(upload_signature([same_name_a]), upload_signature([same_name_b]))

    def test_upload_signature_is_order_insensitive(self):
        file_a = UploadedFileStub("a.pdf", b"a")
        file_b = UploadedFileStub("b.pdf", b"b")
        self.assertEqual(upload_signature([file_a, file_b]), upload_signature([file_b, file_a]))

    def test_upload_signature_keeps_duplicate_files(self):
        file_a = UploadedFileStub("a.pdf", b"a")
        file_b = UploadedFileStub("b.pdf", b"b")
        first = upload_signature([file_a, file_a, file_b])
        second = upload_signature([file_b, file_a, file_a])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)

    def test_upload_signature_uses_full_seekable_content(self):
        file_obj = BytesIO(b"abcdef")
        file_obj.name = "slides.pdf"
        file_obj.read(2)
        current_position = file_obj.tell()

        advanced_pointer_signature = upload_signature([file_obj])
        self.assertEqual(file_obj.tell(), current_position)

        file_obj.seek(0)
        start_pointer_signature = upload_signature([file_obj])
        self.assertEqual(advanced_pointer_signature, start_pointer_signature)


if __name__ == "__main__":
    unittest.main()

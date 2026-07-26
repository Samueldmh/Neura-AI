import os
class MockPoint:
    def __init__(self, _id, score, book_title, text):
        self.id = _id
        self.score = score
        self.payload = {"book_title": book_title, "text": text}

class MockQdrant:
    def query_points(self, collection_name, query, limit):
        pts = [
            MockPoint(1, 0.99, "Robbins Basic Pathology", "Robbins: High score 1"),
            MockPoint(2, 0.95, "Robbins Basic Pathology", "Robbins: High score 2"),
            MockPoint(3, 0.93, "Robbins Basic Pathology", "Robbins: High score 3"),
            MockPoint(4, 0.90, "Robbins Basic Pathology", "Robbins: High score 4"),
            MockPoint(5, 0.88, "Lippincott Illustrated Reviews: Pharmacology", "Lippincott: Medium score 1"),
            MockPoint(6, 0.85, "Lippincott Illustrated Reviews: Pharmacology", "Lippincott: Medium score 2"),
            MockPoint(7, 0.82, "Essentials of Haematology", "Haem: Low score 1"),
        ]
        class Result: pass
        res = Result()
        res.points = pts[:limit]
        return res

import main
main.qdrant = MockQdrant()
main.embedder.embed = lambda text: [[0.1]]

print("--- TEST: Grouping Logic ---")
preferred_books = ["Robbins Basic Pathology", "Lippincott Illustrated Reviews: Pharmacology", "Essentials of Haematology"]
user_msg = "General medical question"
terms = main.extract_medical_terms(user_msg)
res = main.multi_search_qdrant(terms, preferred_books)
for p in res:
    print(f"ID {p.id}: {p.payload['book_title']} -> {p.payload['text']}")

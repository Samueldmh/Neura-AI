import os
import asyncio
from qdrant_client import models

os.environ["QDRANT_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6ZDliOTdhZjYtOTYyOS00YzMxLWJlNTktYTBiZWRkNmQ3NjlhIn0.Oa7xD5-T58Q-r-Az1waWwtwwLVTDi7HgzmtLLJE7cJw"

import main

async def test_match_text():
    print("=== TESTING MATCH_TEXT SEARCH FOR LIPPINCOTT ===")
    query = "Classify antidiabetic drugs"
    query_vector = [e.tolist() for e in main.embedder.embed([query])][0]
    
    # Try querying with MatchText for 'lippincott' or 'pharmacology'
    for test_word in ["Lippincott", "pharmacology", "Robbins", "Haematology"]:
        try:
            hits = main.qdrant.query_points(
                collection_name=main.COLLECTION_NAME,
                query=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="book_title",
                            match=models.MatchText(text=test_word)
                        )
                    ]
                ),
                limit=3
            ).points
            print(f"\nKeyword: '{test_word}' -> Found {len(hits)} hits:")
            for h in hits:
                print(f"  - Book Payload: '{h.payload.get('book_title')}' | Score: {h.score}")
        except Exception as e:
            print(f"Error querying '{test_word}': {e}")

if __name__ == "__main__":
    asyncio.run(test_match_text())

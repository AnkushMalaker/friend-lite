import logging
import time
from pprint import pprint

from qdrant_client import QdrantClient, models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:

    def __init__(self, embed_model, collection_name="qdrant_collection"):
        self.collection_name = collection_name
        self.embed_model = embed_model
        self.qdrant_client = self._set_qdrant_client()

    def _set_qdrant_client(self):
        client = QdrantClient(
            url="http://localhost:6333",
            prefer_grpc=False,
            timeout=1000,
        )
        return client

    def search(self, query, top_k=10):
        query_embedding = next(self.embed_model.embed(query))

        # Start the timer
        start_time = time.time()

        result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            search_params=models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=True,
                    rescore=True,
                    oversampling=2.0,
                )
            ),
            timeout=1000,
        )

        # End the timer
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Log the elapsed time
        logger.info(f"Execution time for the search: {elapsed_time:.4f} seconds")

        return result

    def generate_context(self, query: str) -> str:

        result = self.search(query=query, top_k=2)
        context = [dict(data) for data in result]
        combined_prompt = []

        for entry in context:
            text = entry["payload"]["text"]
            prompt = "" if text is None else text
            combined_prompt.append(prompt)

        return "\n\n---\n\n".join(combined_prompt)

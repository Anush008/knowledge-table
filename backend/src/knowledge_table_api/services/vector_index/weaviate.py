"""Weaviate vector index."""

import logging
from typing import Any, Dict, List

import numpy as np
import weaviate
from langchain.schema import Document
from pydantic_settings import BaseSettings, SettingsConfigDict
from weaviate.classes.query import Filter
from weaviate.datatypes import Int, Text
from weaviate.properties import Property

from knowledge_table_api.config import Settings
from knowledge_table_api.models.query import Chunk, Rule, VectorResponse
from knowledge_table_api.services.llm import decompose_query
from knowledge_table_api.services.llm_service import LLMService
from knowledge_table_api.services.vector_index.base import VectorIndex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WeaviateConfig(BaseSettings):
    """Weaviate configuration."""

    model_config = SettingsConfigDict(env_prefix="WEAVIATE_")

    host: str = "localhost"
    port: int = 8080
    grpc_port: int = 50051


class WeaviateIndex(VectorIndex):
    """Weaviate vector index."""

    def __init__(self) -> None:
        settings = Settings()
        self.collection_name = settings.index_name
        self.dimensions = settings.dimensions
        weaviate_config = WeaviateConfig()
        self.client = weaviate.connect_to_local(
            host=weaviate_config.host,
            port=weaviate_config.port,
            grpc_port=weaviate_config.grpc_port,
        )
        self.ensure_schema_exists()

    def ensure_schema_exists(self) -> None:
        """Ensure the schema exists."""
        if not self.client.collections.exists(self.collection_name):
            self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=None,
                properties=[
                    Property(
                        name="text",
                        data_type=Text(),
                    ),
                    Property(
                        name="page_number",
                        data_type=Int(),
                    ),
                    Property(
                        name="chunk_number",
                        data_type=Int(),
                    ),
                    Property(
                        name="document_id",
                        data_type=Text(),
                    ),
                ],
            )

    async def upsert_vectors(
        self, document_id: str, chunks: List[Document], llm_service: LLMService
    ) -> Dict[str, str]:
        """Upsert the vectors into Weaviate."""
        entries = self.prepare_chunks(document_id, chunks, llm_service)
        logger.info(f"Upserting {len(entries)} chunks")

        with self.client.batch.dynamic() as batch:
            for entry in entries:
                vector = entry.pop("vector")
                batch.add_object(
                    properties=entry,
                    collection=self.collection_name,
                    vector=vector,
                )

        return {"message": f"Successfully upserted {len(entries)} chunks."}

    async def vector_search(
        self, queries: List[str], document_id: str, llm_service: LLMService
    ) -> dict[str, Any]:
        """Perform a vector search."""
        logger.info(f"Retrieving vectors for {len(queries)} queries.")

        embeddings = llm_service.get_embeddings()
        final_chunks: List[Dict[str, Any]] = []

        for query in queries:
            logger.info("Generating embedding.")
            embedded_query = np.array(embeddings.embed_query(query)).tolist()
            logger.info("Searching...")

            collection = self.client.collections.get(self.collection_name)
            query_response = collection.query.near_vector(
                filters=Filter.by_property("document_id").equal(document_id),
                near_vector=embedded_query,
                return_properties=["text", "page_number", "chunk_number"],
                limit=40,
            )

            final_chunks.extend(
                query_response["data"]["Get"][self.collection_name]
            )

        seen_chunks, formatted_output = set(), []

        for chunk in final_chunks:
            if chunk["chunk_number"] not in seen_chunks:
                seen_chunks.add(chunk["chunk_number"])
                formatted_output.append(
                    {"content": chunk["text"], "page": chunk["page_number"]}
                )

        logger.info(f"Retrieved {len(formatted_output)} unique chunks.")
        return {
            "message": "Query processed successfully.",
            "chunks": formatted_output,
        }

    async def hybrid_search(
        self,
        query: str,
        document_id: str,
        rules: list[Rule],
        llm_service: LLMService,
    ) -> VectorResponse:
        """Perform a hybrid search."""
        logger.info("Performing hybrid search.")

        embeddings = llm_service.get_embeddings()

        sorted_keyword_chunks = []
        keywords = await self.extract_keywords(query, rules, llm_service)

        if keywords:
            keyword_filter = {
                "operator": "And",
                "operands": [
                    {
                        "operator": "Or",
                        "operands": [
                            {
                                "path": ["text"],
                                "operator": "Like",
                                "valueString": f"*{keyword}*",
                            }
                            for keyword in keywords
                        ],
                    },
                    {
                        "path": ["document_id"],
                        "operator": "Equal",
                        "valueString": document_id,
                    },
                ],
            }

            logger.info("Running query with keyword filters.")
            collection = self.client.collections.get(self.collection_name)
            keyword_response = collection.query.fetch_objects(
                return_properties=["text", "page_number", "chunk_number"],
                filters=keyword_filter,
            )

            keyword_chunks = keyword_response["data"]["Get"][
                self.collection_name
            ]

            def count_keywords(text: str, keywords: List[str]) -> int:
                return sum(
                    text.lower().count(keyword.lower()) for keyword in keywords
                )

            sorted_keyword_chunks = sorted(
                keyword_chunks,
                key=lambda chunk: count_keywords(
                    chunk["text"], keywords or []
                ),
                reverse=True,
            )

        embedded_query = np.array(embeddings.embed_query(query)).tolist()
        logger.info("Running semantic similarity search.")

        semantic_response = (
            self.client.query.get(
                self.collection_name, ["text", "page_number", "chunk_number"]
            )
            .with_where(
                {
                    "path": ["document_id"],
                    "operator": "Equal",
                    "valueString": document_id,
                }
            )
            .with_near_vector({"vector": embedded_query})
            .with_limit(40)
            .do()
        )

        semantic_chunks = semantic_response["data"]["Get"][
            self.collection_name
        ]

        print(f"Found {len(semantic_chunks)} semantic chunks.")

        # Combine the top results from keyword and semantic searches
        combined_chunks = sorted_keyword_chunks[:20] + semantic_chunks

        # Sort the combined results by chunk number
        combined_sorted_chunks = sorted(
            combined_chunks, key=lambda chunk: chunk["chunk_number"]
        )

        # Eliminate duplicate chunks
        seen_chunks = set()
        formatted_output = []

        for chunk in combined_sorted_chunks:
            if chunk["chunk_number"] not in seen_chunks:
                formatted_output.append(
                    {"content": chunk["text"], "page": chunk["page_number"]}
                )
                seen_chunks.add(chunk["chunk_number"])

        logger.info(f"Retrieved {len(formatted_output)} unique chunks.")

        return VectorResponse(
            message="Query processed successfully.",
            chunks=[Chunk(**chunk) for chunk in formatted_output],
        )

    async def decomposed_search(
        self,
        query: str,
        document_id: str,
        rules: List[Rule],
        llm_service: LLMService,
    ) -> Dict[str, Any]:
        """Perform a decomposed search."""
        logger.info("Decomposing query into smaller sub-queries.")
        decomposition_response = await decompose_query(
            llm_service=llm_service, query=query
        )
        sub_query_chunks = await self.vector_search(
            decomposition_response["sub-queries"], document_id, llm_service
        )
        return {
            "sub_queries": decomposition_response["sub-queries"],
            "chunks": sub_query_chunks["chunks"],
        }

    async def delete_document(self, document_id: str) -> Dict[str, str]:
        """Delete a document from Weaviate."""
        collection = self.client.collections.get(self.collection_name)
        collection.data.delete_many(
            where=Filter.by_property("document_id").equal(document_id)
        )
        return {
            "status": "success",
            "message": "Document deleted successfully.",
        }

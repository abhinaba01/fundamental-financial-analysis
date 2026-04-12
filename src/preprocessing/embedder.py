"""
EmbeddingPipeline: Final preprocessing stage - embeddings and vector store indexing.

Performs:
- Chunk embedding using BAAI/bge-large-en-v1.5
- L2 normalization of vectors
- ChromaDB indexing with metadata
- Retrieval with optional filtering by ticker

Output: ParsedDocument with embeddings populated, vector store updated
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

try:
    import chromadb
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    chromadb = None
    np = None
    SentenceTransformer = None

from src.preprocessing.document import DocumentChunk, ParsedDocument
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configuration
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32
COLLECTION_NAME = "financial_docs"
VECTOR_STORE_PATH = Path("data/vector_store")


class EmbeddingPipeline:
    """Embed document chunks and manage ChromaDB vector store."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        batch_size: int = BATCH_SIZE,
        collection_name: str = COLLECTION_NAME,
        vector_store_path: str | Path = VECTOR_STORE_PATH,
        device: str = "cpu",
    ):
        """
        Initialize the EmbeddingPipeline.

        Args:
            model_name: HuggingFace model ID for embeddings
            batch_size: Number of chunks to embed per forward pass
            collection_name: ChromaDB collection name
            vector_store_path: Path to ChromaDB persistent storage
            device: Device to run model on ('cpu' or 'cuda')
        """
        if chromadb is None or np is None or SentenceTransformer is None:
            raise ImportError(
                "Required packages not installed. "
                "Install with: pip install chromadb sentence-transformers pandas numpy"
            )

        self.logger = logger
        self.model_name = model_name
        self.batch_size = batch_size
        self.collection_name = collection_name
        self.vector_store_path = Path(vector_store_path)
        self.device = device

        # Create vector store directory if it doesn't exist
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client (persistent)
        self.client = chromadb.PersistentClient(
            path=str(self.vector_store_path)
        )

        # Initialize model
        self.logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

        # Initialize or get collection
        self.collection = None
        self._init_collection()

    def _init_collection(self):
        """Initialize or retrieve ChromaDB collection."""
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.logger.info(
                f"Initialized ChromaDB collection: {self.collection_name}"
            )
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB collection: {e}")
            raise

    def embed_and_index(self, doc: ParsedDocument) -> ParsedDocument:
        """
        Embed all chunks in a document and upsert into ChromaDB.

        Args:
            doc: ParsedDocument with chunks populated

        Returns:
            ParsedDocument with chunk.embedding populated for all chunks
        """
        self.logger.info(
            f"Embedding {len(doc.chunks)} chunks from document: {doc.doc_id}"
        )

        if not doc.chunks:
            self.logger.warning("Document has no chunks. Skipping embedding.")
            return doc

        # Extract texts for embedding
        texts = [chunk.text for chunk in doc.chunks]

        # Encode in batches
        embeddings = self._encode_batch(texts)

        # Update chunks with embeddings
        for chunk, embedding in zip(doc.chunks, embeddings):
            chunk.embedding = embedding.tolist()

        # Upsert to ChromaDB
        self._upsert_to_chromadb(doc)

        self.logger.info(f"Embedded and indexed {len(doc.chunks)} chunks")

        return replace(doc, chunks=doc.chunks)

    def _encode_batch(self, texts: list[str]):
        """
        Encode texts to embeddings using batch processing.

        Args:
            texts: List of text strings to encode

        Returns:
            NumPy array of shape (len(texts), embedding_dim), L2-normalized
        """
        embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            self.logger.debug(f"Encoding batch {i // self.batch_size + 1}")

            batch_embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            embeddings.append(batch_embeddings)

        # Concatenate all batches
        if embeddings:
            import numpy as np
            all_embeddings = np.vstack(embeddings)
        else:
            import numpy as np
            all_embeddings = np.array([])

        return all_embeddings

    def _upsert_to_chromadb(self, doc: ParsedDocument):
        """
        Upsert document chunks to ChromaDB collection.

        Args:
            doc: ParsedDocument with embedded chunks
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            ids = []
            embeddings = []
            documents = []
            metadatas = []

            for chunk in doc.chunks:
                if chunk.embedding is None:
                    self.logger.warning(f"Chunk {chunk.chunk_id} has no embedding. Skipping.")
                    continue

                ids.append(chunk.chunk_id)
                embeddings.append(chunk.embedding)
                documents.append(chunk.text)

                # Build metadata for this chunk
                metadata = {
                    "doc_id": doc.doc_id,
                    "ticker": doc.ticker or "unknown",
                    "chunk_type": chunk.chunk_type.value,
                    "fiscal_period": doc.fiscal_period or "unknown",
                    "page_number": chunk.page_number or -1,
                }
                metadata.update(chunk.metadata or {})
                metadatas.append(metadata)

            if ids:
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas,
                )
                self.logger.debug(
                    f"Upserted {len(ids)} chunks to ChromaDB"
                )

        except Exception as e:
            self.logger.error(f"Error upserting to ChromaDB: {e}")
            raise

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_ticker: str | None = None,
        similarity_threshold: float = 0.75,
    ) -> list[DocumentChunk]:
        """
        Retrieve top-k most similar chunks for a query.

        Args:
            query: Query text
            top_k: Number of chunks to retrieve
            filter_ticker: Optional ticker symbol to filter by
            similarity_threshold: Minimum cosine similarity score

        Returns:
            List of DocumentChunk objects sorted by similarity (descending)
        """
        if not self.collection:
            raise RuntimeError("ChromaDB collection not initialized")

        try:
            # Encode query
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).tolist()

            # Build where filter if ticker specified
            where_filter = None
            if filter_ticker:
                where_filter = {"ticker": {"$eq": filter_ticker}}

            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["distances", "documents", "metadatas"],
            )

            # Convert results to DocumentChunk objects
            retrieved_chunks = []

            if results and results["documents"]:
                for doc, distance, metadata in zip(
                    results["documents"][0],
                    results["distances"][0],
                    results["metadatas"][0],
                ):
                    # ChromaDB returns distance, convert to similarity
                    # For cosine, similarity = 1 - distance
                    similarity = 1 - distance

                    if similarity >= similarity_threshold:
                        chunk = DocumentChunk(
                            chunk_id=metadata.get("chunk_id", "unknown"),
                            text=doc,
                            chunk_type=metadata.get("chunk_type", "general"),
                            page_number=metadata.get("page_number"),
                            metadata={
                                "similarity": float(similarity),
                                **metadata,
                            },
                        )
                        retrieved_chunks.append(chunk)

            self.logger.debug(
                f"Retrieved {len(retrieved_chunks)} chunks for query "
                f"(threshold={similarity_threshold})"
            )

            return retrieved_chunks

        except Exception as e:
            self.logger.error(f"Error retrieving from ChromaDB: {e}")
            raise

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
            self._init_collection()
            self.logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            self.logger.warning(f"Error clearing collection: {e}")

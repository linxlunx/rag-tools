from .base import Base
from sqlalchemy import Column, Integer, Text, TIMESTAMP, func, Index
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_index = Column(Integer, nullable=False)
    original_text = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    contextualized_text = Column(Text, nullable=False)

    # 768-dimension vector (Gemini text-embedding-004)
    embedding = Column(Vector(768))


    meta = Column("metadata", JSONB)
    created_at = Column(TIMESTAMP, server_default=func.now())

    # --- Indexes (matching your SQL) ---
    __table_args__ = (
        # vector similarity search (ivfflat + cosine)
        Index(
            "chunks_embedding_idx",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"lists": "100"}
        ),
        
        # JSONB GIN index
        Index(
            "chunks_metadata_idx",
            "metadata",
            postgresql_using="gin"
        ),
    )
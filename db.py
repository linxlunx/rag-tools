from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
from models.chunk import Chunk
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = None
        self.session_factory = None
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize database engine and session factory"""
        self.engine = create_engine(self.db_url, echo=False)
        self.session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_session(self) -> Session:
        """Create and return a new database session"""
        if self.session_factory is None:
            self._initialize_engine()
        return self.session_factory()
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions with automatic cleanup"""
        session = self.create_session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

class DBSession:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.engine = None
        self.session = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and session"""
        self.engine = create_engine(self.db_url, echo=False)
        Session = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.session = Session()
    
    def store_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """
        Store a single chunk in the database.
        
        Args:
            chunk_data: Dictionary containing chunk data
        """
        if self.session is None:
            raise RuntimeError("Session not initialized. Call _initialize() first.")
        
        chunk = Chunk(**chunk_data)
        self.session.add(chunk)
    
    def store_chunks(self, chunks_data: List[Dict[str, Any]]) -> None:
        """
        Store multiple chunks in the database.
        
        Args:
            chunks_data: List of dictionaries containing chunk data
        """
        if self.session is None:
            raise RuntimeError("Session not initialized. Call _initialize() first.")
        
        chunks = [Chunk(**chunk_data) for chunk_data in chunks_data]
        self.session.add_all(chunks)
    
    def commit(self):
        """Commit the current transaction"""
        if self.session is None:
            raise RuntimeError("Session not initialized. Call _initialize() first.")
        
        try:
            self.session.commit()
        except Exception as e:
            self.session.rollback()
            raise e
    
    def rollback(self):
        """Rollback the current transaction"""
        if self.session is None:
            raise RuntimeError("Session not initialized. Call _initialize() first.")
        
        self.session.rollback()
    
    def close(self):
        """Close the database session"""
        if self.session:
            self.session.close()
            self.session = None
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup"""
        if exc_type is not None:
            self.rollback()
        else:
            try:
                self.commit()
            except Exception:
                self.rollback()
                raise
    
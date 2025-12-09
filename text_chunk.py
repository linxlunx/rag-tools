import google.generativeai as genai
from typing import List, Dict, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import json

class GeminiContextualRetrieval:
    """
    Contextual Retrieval implementation using Gemini for both:
    - LLM: Context generation
    - Embeddings: Vector embeddings
    """
    
    def __init__(
        self,
        google_api_key: str,
        google_model: str,
        google_embedding_model: str,
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ):
        genai.configure(api_key=google_api_key)
        self.gemini_model = genai.GenerativeModel(google_model)
        self.google_embedding_model = google_embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: str, doc_metadata: Dict = None) -> List[Dict]:
        """
        Split document into overlapping chunks.
        
        Args:
            document: Full document text
            doc_metadata: Optional metadata (title, source, date, etc.)
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        words = document.split()
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_index': len(chunks),
                'metadata': doc_metadata or {}
            })
            
        return chunks

    def ask_gemini(self, prompt: str) -> str:
        """
        Use Gemini LLM to generate content based on a prompt.
        
        Args:
            prompt: The prompt string to send to Gemini
            
        Returns:
            Generated text response from Gemini
        """
        response = self.gemini_model.generate_content(prompt)
        return response.text.strip()
    
    def generate_context_for_chunk(
        self,
        whole_document: str,
        chunk_text: str
    ) -> str:
        """
        Use Gemini to generate contextual information for a single chunk.
        
        Args:
            whole_document: The complete document text
            chunk_text: The specific chunk to contextualize
            
        Returns:
            Contextual description string (50-100 tokens)
        """
        prompt = f"""<document>
{whole_document}
</document>

Here is the chunk we want to situate within the whole document
<chunk>
{chunk_text}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""
        
        response = self.ask_gemini(prompt)
        return response
    
    def process_document(self, document: str, doc_metadata: Dict = None) -> List[Dict]:
        """
        Complete preprocessing pipeline for a single document:
        1. Chunk the document
        2. Generate context for each chunk
        3. Prepend context to original chunk
        
        Args:
            document: Full document text
            doc_metadata: Optional metadata
            
        Returns:
            List of processed chunks with contextualized text
        """
        print(f"Chunking document...")
        chunks = self.chunk_document(document, doc_metadata)
        
        print(f"Generated {len(chunks)} chunks. Generating contextual information with Gemini...")
        
        contextualized_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            
            # Generate context
            context = self.generate_context_for_chunk(
                whole_document=document,
                chunk_text=chunk['text']
            )
            
            # Create contextualized version
            contextualized_text = f"{context}\n\n{chunk['text']}"
            
            contextualized_chunks.append({
                'original_text': chunk['text'],
                'context': context,
                'contextualized_text': contextualized_text,
                'chunk_index': chunk['chunk_index'],
                'metadata': chunk['metadata']
            })
            
        return contextualized_chunks
    
    def create_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Gemini.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for contextualized chunks using Gemini.
        
        Args:
            chunks: List of chunk dictionaries with 'contextualized_text'
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk['contextualized_text'] for chunk in chunks]
        
        print(f"Generating Gemini embeddings for {len(texts)} chunks...")
        
        # Gemini embedding API
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0:
                print(f"  Embedding chunk {i+1}/{len(texts)}...")
            
            result = genai.embed_content(
                model=self.google_embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])
        
        return np.array(embeddings)
    
    def create_bm25_index(self, chunks: List[Dict]) -> BM25Okapi:
        """
        Create BM25 index from contextualized chunks for exact term matching.
        
        Args:
            chunks: List of chunk dictionaries with 'contextualized_text'
            
        Returns:
            BM25Okapi index object
        """
        print(f"Creating BM25 index...")
        
        # Tokenize the contextualized text
        tokenized_chunks = [
            chunk['contextualized_text'].lower().split()
            for chunk in chunks
        ]
        
        bm25_index = BM25Okapi(tokenized_chunks)
        return bm25_index
    
    def preprocess_knowledge_base(
        self,
        documents: List[Tuple[str, Dict]]
    ) -> Dict:
        """
        Complete preprocessing pipeline for entire knowledge base.
        
        Args:
            documents: List of (document_text, metadata) tuples
            
        Returns:
            Dictionary containing all processed data:
            - chunks: List of all contextualized chunks
            - embeddings: Numpy array of embeddings
            - bm25_index: BM25 index object
        """
        all_chunks = []
        
        for i, (doc_text, doc_metadata) in enumerate(documents):
            print(f"\n=== Processing document {i+1}/{len(documents)} ===")
            chunks = self.process_document(doc_text, doc_metadata)
            all_chunks.extend(chunks)
        
        print(f"\nTotal chunks processed: {len(all_chunks)}")
        
        # Create embeddings
        embeddings = self.create_embeddings(all_chunks)

        print(all_chunks)
        
        # Create BM25 index
        bm25_index = self.create_bm25_index(all_chunks)
        
        return {
            'chunks': all_chunks,
            'embeddings': embeddings,
            'bm25_index': bm25_index
        }
    
    def save_preprocessed_data(self, data: Dict, output_path: str):
        """
        Save preprocessed data to disk for later use.
        
        Args:
            data: Preprocessed data dictionary
            output_path: Base path for output files
        """
        # Save embeddings
        np.save(f"{output_path}_embeddings.npy", data['embeddings'])
        
        # Save chunks
        with open(f"{output_path}_chunks.json", 'w') as f:
            json.dump(data['chunks'], f, indent=2)
        
        print(f"Saved preprocessed data to {output_path}")

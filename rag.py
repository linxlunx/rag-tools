from text_chunk import GeminiContextualRetrieval
from db import DBSession
from models.chunk import Chunk

class RAG:
    def __init__(self, config):
        self.config = config
        self.preprocessor = GeminiContextualRetrieval(
            google_api_key=self.config.GEMINI_API_KEY,
            google_model=self.config.GEMINI_MODEL,
            google_embedding_model=self.config.GEMINI_EMBEDDING_MODEL,
            chunk_size=800,
            chunk_overlap=100)

    def process_documents(self, raw_documents):
        documents = []
        for page_text in raw_documents:
            documents.append((page_text[0], page_text[1]))
        
        preprocessed_data = self.preprocessor.preprocess_knowledge_base(documents)

        with DBSession(self.config.DATABASE_URL) as session:
            for (c, e) in zip(
                preprocessed_data['chunks'],
                preprocessed_data['embeddings']):
                db_chunk = {
                    "chunk_index": c['chunk_index'],
                    "original_text": c['original_text'],
                    "context": c['context'],
                    "contextualized_text": c['contextualized_text'],
                    "embedding": e,
                    "metadata": c['metadata']
                }
                session.store_chunk(db_chunk)
            session.commit()
        
        print("Successfully processed and stored chunks in the database.")

    def search_similar(self, query: str, top_k: int = 5):
        query_embedding = self.preprocessor.create_embedding([query])[0]

        with DBSession(self.config.DATABASE_URL) as session:
            results = session.session.query(
                Chunk,
                (1 - Chunk.embedding.cosine_distance(query_embedding)).label('similarity')
            ).order_by(
                Chunk.embedding.cosine_distance(query_embedding)
            ).limit(top_k).all()
            
            return [
                {
                    'id': chunk.id,
                    'chunk_index': chunk.chunk_index,
                    'original_text': chunk.original_text,
                    'context': chunk.context,
                    'contextualized_text': chunk.contextualized_text,
                    'metadata': chunk.meta,
                    'similarity': float(similarity),
                    'created_at': chunk.created_at
                }
                for chunk, similarity in results
            ]
        
        return results

    def ask_llm(self, question: str, contexts: list) -> str:
        context = "\n\n".join(
            f"Chunk {c['chunk_index']}:\n{c['contextualized_text']}"
            for c in contexts
        )

        prompt = f"""
            You are an expert assistant. Answer the question using ONLY the context provided below.
            If the answer is not in the context, say "I don't know based on the given information."

            Context:
            {context}

            Question:
            {question}
        """

        answer = self.preprocessor.ask_gemini(prompt)
        return answer
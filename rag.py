from config import Config
from text_chunk import GeminiContextualRetrieval
from pdf_parser import PdfParser
import click
import sys

@click.command()
@click.option('--pdf-file', type=click.Path(exists=True), help='Path to the PDF file to process.')


def main(pdf_file):
    config = Config()
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    
    # Initialize with Gemini API key
    preprocessor = GeminiContextualRetrieval(
        google_api_key=config.GEMINI_API_KEY,
        google_model=config.GEMINI_MODEL,
        google_embedding_model=config.GEMINI_EMBEDDING_MODEL,
        chunk_size=800,
        chunk_overlap=100
    )
    

    pdf_parser = PdfParser(file_path=pdf_file)

    # Read and combine all pages from the PDF
    documents = []
    for page_text in pdf_parser.read_pages_generator():
        documents.append((page_text, {"source": pdf_file}))
    
    # Preprocess the knowledge base
    preprocessed_data = preprocessor.preprocess_knowledge_base(documents)
    
    # Save for later use
    preprocessor.save_preprocessed_data(preprocessed_data, "knowledge_base")
    
    # Inspect results
    print(f"\n{'='*60}")
    print(f"PREPROCESSING RESULTS")
    print(f"{'='*60}")
    print(f"Number of chunks: {len(preprocessed_data['chunks'])}")
    print(f"Embedding shape: {preprocessed_data['embeddings'].shape}")
    print(f"\nExample contextualized chunk:")
    print(f"{'-'*60}")
    print(preprocessed_data['chunks'][0]['contextualized_text'][:500])
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
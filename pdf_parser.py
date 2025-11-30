from pypdf import PdfReader
from typing import Generator

class PdfParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def read_pages_chunked(self, chunk_size: int = 1) -> Generator[str, None, None]:
        """
        Generator that yields text from PDF pages in chunks.
        Creates a new PdfReader instance to minimize memory usage.
        
        Args:
            chunk_size: Number of pages to process at once (default: 1)
        
        Yields:
            Text content from chunk_size pages at a time
        """
        with PdfReader(self.file_path) as reader:
            page_buffer = []
            total_pages = len(reader.pages)
            
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                page_buffer.append(page_text)
                
                # Yield when we reach chunk_size or at the last page
                if len(page_buffer) == chunk_size or i == total_pages - 1:
                    yield "\n".join(page_buffer) + "\n"
                    page_buffer.clear()
    
    def read_pages_generator(self) -> Generator[str, None, None]:
        """
        Generator that yields text from each PDF page individually.
        Creates a new PdfReader instance to minimize memory usage.
        
        Yields:
            Text content from one page at a time
        """
        with PdfReader(self.file_path) as reader:
            for page in reader.pages:
                yield page.extract_text() + "\n"
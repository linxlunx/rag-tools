from config import Config
from text_chunk import GeminiContextualRetrieval

def main():
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
    
    # Example documents
    documents = [
        (
            """ACME Corporation Q2 2023 Financial Report
            
            Executive Summary:
            ACME Corporation demonstrated strong performance in Q2 2023, with revenue 
            reaching $314 million. The company's revenue grew by 3% over the previous 
            quarter, driven primarily by increased sales in the enterprise software 
            division.
            
            Product Performance:
            Our flagship product, ACME Cloud Suite, saw a 15% increase in adoption 
            among Fortune 500 companies. The AI-powered analytics module was particularly 
            well-received, contributing $45 million in additional revenue.
            
            Market Outlook:
            Looking ahead to Q3 2023, we anticipate continued growth in the cloud 
            infrastructure segment. Industry analysts project a 20% market expansion 
            in the enterprise software sector, and ACME is well-positioned to capture 
            significant market share.
            """,
            {
                "source": "SEC Filing",
                "company": "ACME Corp",
                "period": "Q2 2023",
                "document_type": "financial_report"
            }
        ),
        (
            """ACME Corporation Product Guide
            
            ACME Cloud Suite Overview:
            The ACME Cloud Suite is an enterprise-grade software platform designed 
            for Fortune 500 companies. It includes modules for data analytics, 
            workflow automation, and security compliance.
            
            Key Features:
            - AI-powered analytics with real-time insights
            - Automated workflow management
            - Enterprise-grade security with SOC 2 compliance
            - Seamless integration with existing systems
            
            Pricing:
            Enterprise licenses start at $50,000 per year with volume discounts 
            available for multi-year contracts.
            """,
            {
                "source": "Product Documentation",
                "company": "ACME Corp",
                "document_type": "product_guide"
            }
        )
    ]
    
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
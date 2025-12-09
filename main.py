import click
from config import Config
from rag import RAG

@click.command()
@click.option('--process-docs', is_flag=True, help='Process and store documents in the database.')
@click.option('--ask', type=str, help='Search for similar chunks to the given query.')
def main(process_docs, ask):
    config = Config()
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
    
    raw_documents = [
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
            for Fortune 500 companies.
            
            Key Features:
            - AI-powered analytics with real-time insights
            - Automated workflow management
            - Enterprise-grade security with SOC 2 compliance
            """,
            {
                "source": "Product Documentation",
                "company": "ACME Corp",
                "document_type": "product_guide"
            }
        )
    ]

    rag_processor = RAG(config)
    if process_docs:
        rag_processor.process_documents(raw_documents)
    if ask:
        contexts = rag_processor.search_similar(ask, top_k=5)
        answer = rag_processor.ask_llm(ask, contexts)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
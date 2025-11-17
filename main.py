from src.stock_data_collector import collect_stock_data
from src.news_data_collector import collect_from_kaggle_folder
from src.sentiment_analyzer import analyze_sentiment
from src.data_merger import merge_data
from src.model_trainer import train_models
from src.vector_db_setup import setup_vector_db
from src.embedding_generator import generate_embeddings
from src.llm_summarizer import summarize_financial_data

def main():
    print("Starting data collection...")
    #collect_stock_data()

    print("Collecting news data...")
    #collect_from_kaggle_folder()

    print("Analyzing sentiment...")
    #analyze_sentiment()
    
    print("Merging data...")
    #merge_data()
    
    print("Training models...")
    #train_models()
    
    print("Setting up vector DB...")
    #setup_vector_db()
    
    print("Generating embeddings...")
    #generate_embeddings()
    
    #print("Pipeline completed.")
    
    # Example summarization with enhanced financial analysis
    print("Testing enhanced financial analysis...")
    
    try:
        summary = summarize_financial_data('RELIANCE.NS')
        print("\n" + "="*50)
        print("FINANCIAL ANALYSIS RESULT:")
        print("="*50)
        print(summary)
    except Exception as e:
        print(f"Error in analysis: {e}")
        
    # Test with multiple companies
    test_companies = ['TCS.NS', 'HDFCBANK.NS']
    for company in test_companies:
        try:
            summary = summarize_financial_data(company)
            print(f"\n{'='*50}")
            print(f"ANALYSIS FOR {company}")
            print(f"{'='*50}")
            print(summary[:500] + "..." if len(summary) > 500 else summary)
        except Exception as e:
            print(f"Error analyzing {company}: {e}")

if __name__ == "__main__":
    main()
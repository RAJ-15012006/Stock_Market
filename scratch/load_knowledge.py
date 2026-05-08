import os
from dotenv import load_dotenv
from financial_agent import knowledge_base

load_dotenv()

def index_pdf():
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in .env")
        print("Please add GOOGLE_API_KEY=your_key_here to your .env file.")
        return

    print("🚀 Starting PDF indexing...")
    try:
        # Load the knowledge base (this will parse the PDF and save to LanceDB)
        knowledge_base.load(recreate=True, upsert=True)
        print("✅ Knowledge base loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")

if __name__ == "__main__":
    index_pdf()

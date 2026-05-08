import os
from dotenv import load_dotenv
from financial_agent import knowledge_base

load_dotenv()

def index_pdf():
    # Force clean start if dimensions are wrong
    import shutil
    if os.path.exists("tmp/lancedb"):
        print("🧹 Cleaning old vector database...")
        shutil.rmtree("tmp/lancedb")

    print("🚀 Starting PDF indexing with FastEmbed...")
    try:
        # Load the knowledge base
        knowledge_base.load(recreate=True, upsert=True)
        print("✅ Knowledge base loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading knowledge base: {e}")

if __name__ == "__main__":
    index_pdf()

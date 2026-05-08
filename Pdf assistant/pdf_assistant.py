# agents which will be interacting with my vector database
# pdf url knowledge - read pdf from url convert into vector embeddings and store in vector database
import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase, PDFKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.embedder.ollama import OllamaEmbedder

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os
("GROQ_API_KEY")

from phi.llm.groq import Groq

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledge_base = PDFKnowledgeBase(
    path="Stock_Market_Master_Guide_150_Pages.pdf",
    vector_db=PgVector2(
        collection="recipes",
        db_url=db_url,
        embedder=OllamaEmbedder(model="nomic-embed-text", dimensions=768)
    )
)

knowledge_base.load()

storage = PgAssistantStorage(
    table_name="pdf_assistant",
    db_url=db_url
)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)

        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]

    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        llm=Groq(model="llama-3.3-70b-versatile"),
        knowledge_base=knowledge_base,
        storage=storage,

        # Show tool calls in the response
        show_tool_calls=True,

        # Enable the assistant to search the knowledge base
        search_knowledge=True,

        # Enable the assistant to read the chat history
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)
import os  # to store/read environment variables.
from dotenv import load_dotenv # to store/read environment variables.
from phi.agent import Agent 
from phi.playground import Playground, serve_playground_app # Creates web-based chat UI.,, start websitestart website
from financial_agent import web_search_agent, finance_agent #  import agents (THIS is why separation matters) f

load_dotenv() # Load environment variables from .env file, like API keys for Groq, Yahoo Finance, etc.
# Set Groq key
if not os.getenv("GROQ_API_KEY"):
    print("Warning: GROQ_API_KEY not found in environment.")

#  Manager Agent (controls others)
multi_ai_agent = Agent( 
    team=[web_search_agent, finance_agent], 
    instructions=[ 
    "Always include sources", "Use tables when needed"],
    show_tool_calls=True,
     markdown=True, )

"""This one:

coordinates
routes tasks
combines responses"""

app = Playground(agents=[multi_ai_agent]).get_app() #  Playground UI
if __name__ == "__main__": 
    serve_playground_app("playground:app", reload=True)


    """
    User
 ↓
Playground UI
 ↓
multi_ai_agent (manager)
 ↓
Chooses specialist agent
 ↓
Tool call happens
 ↓
LLM formats answer
 ↓
Response shown in UI

    """
import pytest
from financial_agent import finance_agent, web_search_agent

def test_finance_agent_initialization():
    assert finance_agent.name == "Finance Agent"
    assert "YFinanceTool" in str(finance_agent.tools)

def test_web_agent_initialization():
    assert web_search_agent.name == "Web Search Agent"
    assert "DuckDuckGo" in str(web_search_agent.tools)

@pytest.mark.asyncio
async def test_agent_team_logic():
    from financial_agent import agent_team
    # Basic check to see if team is configured
    assert len(agent_team.team) == 3

import asyncio
import sys
from pathlib import Path

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from models import model

SERVER = Path(__file__).parent / "m1.6_dice_server.py"


async def main():
    client = MultiServerMCPClient({
        "dice": {
            "transport": "stdio",
            "command": sys.executable,  # the Python running this script
            "args": [str(SERVER)],
        }
    })
    tools = await client.get_tools()  # starts the server and asks which tools it has
    print("Tools from the server:", [t.name for t in tools])

    agent = create_deep_agent(model=model, tools=tools)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Roll a 20-sided die for me."}]}
    )
    print("Final answer:", result["messages"][-1].text)


asyncio.run(main())

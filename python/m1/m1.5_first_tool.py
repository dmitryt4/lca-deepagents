import random

from deepagents import create_deep_agent
from langchain_core.tools import tool

from models import model


@tool
def roll_dice(sides: int = 6) -> int:
    """Roll a die with the given number of sides and return the result."""
    return random.randint(1, sides)


agent = create_deep_agent(
    model=model,
    tools=[roll_dice],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Roll a 20-sided die for me."}]}
)

for message in result["messages"]:
    for call in getattr(message, "tool_calls", None) or []:
        print(f"Model asked for: {call['name']}({call['args']})")
    if message.type == "tool":
        print(f"Tool returned:   {message.content}")
print(f"Final answer:    {result['messages'][-1].text}")

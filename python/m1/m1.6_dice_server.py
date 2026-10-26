import random

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("dice", log_level="WARNING")  # hide per-request logs


@mcp.tool()
def roll_dice(sides: int = 6) -> int:
    """Roll a die with the given number of sides and return the result."""
    return random.randint(1, sides)  # the same tool as Lesson 1.5


if __name__ == "__main__":
    mcp.run()  # start the server and wait for a client

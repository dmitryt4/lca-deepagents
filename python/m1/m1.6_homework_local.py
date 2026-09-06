# python/m1/m1.6_homework.py
"""M1.6 Homework: Connect to a Different MCP Server.

THE IDEA
Lab 1 connected to the LangChain docs MCP server and filtered its tools
down to just search_docs_by_lang_chain. This homework asks you to connect
to a different public MCP server entirely, one that requires no auth
beyond what your labs already use, and put one of its tools to work.

Don't know where to look? A few free, no-auth public servers to try:
  - DeepWiki (https://mcp.deepwiki.com/mcp): ask questions about any
    public GitHub repo's code and docs.
  - X Docs (https://docs.x.com/mcp): search and retrieve X's public API
    documentation.
Or find your own!

WHAT YOU FILL IN
  TODO 1: build and return the filtered list of MCP tools to use from a
    server and filter of your own choosing.
  TODO 2: write a question suited to your chosen server's own domain,
    not Lab 1's "what is MCP..." question, which won't make sense to
    ask a server about GitHub repos, API docs, or whatever you picked.

RUN
  cd python
  uv run ./m1/m1.6_homework.py
"""

import asyncio
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

from models import model


# ════════════════════════════════════════════════════════════════════════
# TODO 1: Build your MCP client and return its filtered tool list.
#
# Requirements:
#   - Point "url" at a different public MCP server than Lab 1's
#     docs-langchain server (no auth/API key required beyond what your
#     labs already use). See the module docstring for two ready-to-use
#     options.
#   - Fetch tools with client.get_tools() and filter them the same way
#     Lab 1 did, with an ALLOWED set.
#
# Example shape (delete this and write your own):
#   async def build_tools():
#       client = MultiServerMCPClient({
#           "my-server": {"transport": "http", "url": "https://..."}
#       })
#       tools = await client.get_tools()
#       ALLOWED = {"some_tool_name"}
#       return [t for t in tools if t.name in ALLOWED]
# ════════════════════════════════════════════════════════════════════════

async def build_tools():
    client = MultiServerMCPClient({
        "google-workspace": {
          "transport": "stdio",
          "command": "uvx",
          "args": [
            "workspace-mcp",
            "--tools", "sheets", "drive",
          ],
          "env": {
            "PATH": os.environ.get("PATH", ""),
            "GOOGLE_OAUTH_CLIENT_ID": os.environ.get("client_id", ""),
            "GOOGLE_OAUTH_CLIENT_SECRET": os.environ.get("client_secret", ""),
          }
        }
    })
    tools = await client.get_tools()

    ALLOWED = (
        # Google Sheets Tools
        "get_spreadsheet",
        "get_values",
        "read_values",
        "update_values",
        "write_values",
        "append_values",
        "clear_values",
        "create_spreadsheet",
        "update_formulas",
        "insert_dimension",
        "import_to_google_sheets",
        "read_sheet_comments",
        # Google Drive Tools
        "search_drive_files",
        "list_files",
        "get_drive_file",
        "read_file",
        "create_drive_file",
        "create_drive_folder",
        "update_drive_file",
        "copy_drive_file",
        "manage_drive_access",
        "delete_drive_file",
    )
    return [t for t in tools if t.name in ALLOWED]
    

# ════════════════════════════════════════════════════════════════════════
# TODO 2: Write a question suited to your chosen server's own domain,
# not Lab 1's "what is MCP..." question.
# ════════════════════════════════════════════════════════════════════════

QUESTION = (
    "Use Google Drive and Google Sheets tools to find my spreadsheets "
    "and summarize the contents of the most recent one."
)


async def main():
    tools = await build_tools()
    print(f"Loaded {len(tools)} filtered tool(s):")
    for t in tools:
        print(f"  - {t.name}")
    agent = create_deep_agent(model=model, tools=tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})
    print(result["messages"][-1].content)


asyncio.run(main())

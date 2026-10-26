# python/m1/m1.6_homework_local.py
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
  uv run ./m1/m1.6_homework_local.py
"""

import asyncio
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from models import model

ALLOWED = {
    # Google Sheets Tools
    "list_spreadsheets",
    "get_spreadsheet_info",
    "read_sheet_values",
    "create_spreadsheet",
    "create_sheet",
    "list_spreadsheet_comments",
    "import_to_google_sheets",
    # Google Drive Tools
    "search_drive_files",
    "get_drive_file_content",
    "get_drive_file_download_url",
    "list_drive_items",
    "create_drive_file",
    "create_drive_folder",
    "update_drive_file",
    "copy_drive_file",
    "manage_drive_access",
}

USER_EMAIL = os.environ.get("USER_GOOGLE_EMAIL", "dmitryt4@gmail.com")

QUESTION = (
    f"Use Google Drive and Google Sheets tools (for user {USER_EMAIL}) "
    "to find the spreadsheet named 'סידור אפריל 2026' "
    "and try to understand what it is about."
)


def build_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "google-workspace": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "workspace-mcp",
                "--single-user",
                "--tools", "sheets", "drive",
            ],
            "env": {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "USER_GOOGLE_EMAIL": USER_EMAIL,
                "GOOGLE_OAUTH_CLIENT_ID": os.environ.get("GOOGLE_OAUTH_CLIENT_ID") or os.environ.get("client_id", ""),
                "GOOGLE_OAUTH_CLIENT_SECRET": os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET") or os.environ.get("client_secret", ""),
            },
        }
    })


async def main():
    client = build_client()

    # Keep the stdio session open so localhost:8000 stays alive during OAuth
    async with client.session("google-workspace") as session:
        all_tools = await load_mcp_tools(session)
        tools = [t for t in all_tools if t.name in ALLOWED]

        print(f"Loaded {len(tools)} filtered tool(s):")
        for t in tools:
            print(f"  - {t.name}")

        agent = create_deep_agent(model=model, tools=tools)
        result = await agent.ainvoke({"messages": [{"role": "user", "content": QUESTION}]})

        last_msg = str(result["messages"][-1].content)
        if any(word in last_msg.lower() for word in ("authorize", "authentication", "oauth")):
            print("\n--- Action Required ---")
            print(last_msg)
            input("\nComplete the Google login in your browser, then press ENTER to continue...")
            result = await agent.ainvoke({
                "messages": [
                    *result["messages"],
                    {"role": "user", "content": "Authorization is complete. Please proceed."},
                ]
            })

        print("\n--- Response ---")
        print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

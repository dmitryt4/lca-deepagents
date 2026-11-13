"""Deep Agent connected to Google Sheets.

Gives a deep agent tools to list, read, search, append to, and update a
Google Sheets spreadsheet. Every write pauses for your approval first
(human-in-the-loop), so the agent can't change your sheet without you.

Setup (one time):
  1. In Google Cloud Console, create a project and enable the
     "Google Sheets API" (and "Google Drive API" if you want to open sheets by name).
  2. Create a Service Account -> Keys -> Add key -> JSON. Save the file,
     e.g. python/sheets/service_account.json (it's gitignored below; never commit it).
  3. Open your spreadsheet in the browser -> Share -> add the service account's
     email (client_email in the JSON, ...@...iam.gserviceaccount.com) as Editor.
  4. Add to python/.env:
       GOOGLE_SERVICE_ACCOUNT_FILE=sheets/service_account.json
       GOOGLE_SHEET_ID=<the long id from the sheet URL: /spreadsheets/d/<ID>/edit>

Run (from the python/ directory):
  uv run python sheets/sheets_agent.py
  uv run python sheets/sheets_agent.py "Summarize the Sales tab by region"
"""

import json
import os
import sys
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find models.py

import gspread
from deepagents import create_deep_agent
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from models import model  # also loads python/.env

PYTHON_DIR = Path(__file__).resolve().parent.parent
MAX_ROWS_RETURNED = 200  # keep tool output small enough for the model's context


# ── Google Sheets connection ─────────────────────────────────────────────────


def _open_spreadsheet() -> gspread.Spreadsheet:
    creds_path = os.environ.get(
        "GOOGLE_SERVICE_ACCOUNT_FILE", "sheets/service_account.json"
    )
    creds = Path(creds_path)
    if not creds.is_absolute():
        creds = PYTHON_DIR / creds
    if not creds.exists():
        sys.exit(
            f"Service account file not found: {creds}\nSee the setup notes at the top of this file."
        )

    sheet_id = os.environ.get("GOOGLE_SHEET_ID")
    if not sheet_id:
        sys.exit("Set GOOGLE_SHEET_ID in python/.env (the id from the sheet's URL).")

    client = gspread.service_account(filename=str(creds))
    return client.open_by_key(sheet_id)


spreadsheet = _open_spreadsheet()


def _ws(worksheet: str) -> gspread.Worksheet:
    return spreadsheet.worksheet(worksheet)


def _as_table(rows: list[list], start_row: int = 1) -> str:
    """Render rows as numbered, tab-separated lines so the agent knows row numbers."""
    if not rows:
        return "(empty)"
    shown = rows[:MAX_ROWS_RETURNED]
    lines = [
        f"{start_row + i}\t" + "\t".join(str(c) for c in row)
        for i, row in enumerate(shown)
    ]
    if len(rows) > len(shown):
        lines.append(
            f"... {len(rows) - len(shown)} more rows not shown; read a narrower range."
        )
    return "\n".join(lines)


# ── Tools ────────────────────────────────────────────────────────────────────


@tool
def list_worksheets() -> str:
    """List the tabs in the spreadsheet with their size and header row."""
    out = [f"Spreadsheet: {spreadsheet.title}"]
    for ws in spreadsheet.worksheets():
        header = ws.row_values(1)
        out.append(
            f"- {ws.title!r}: {ws.row_count} rows x {ws.col_count} cols; header: {header}"
        )
    return "\n".join(out)


@tool
def read_range(worksheet: str, a1_range: str = "") -> str:
    """Read cells from a tab. a1_range is A1 notation like 'A1:D20' or 'A:C';
    leave it empty to read the whole tab. Each output line starts with its sheet row number."""
    try:
        ws = _ws(worksheet)
        if a1_range:
            rows = ws.get(a1_range)
            start = int(
                "".join(ch for ch in a1_range.split(":")[0] if ch.isdigit()) or 1
            )
        else:
            rows, start = ws.get_all_values(), 1
        return _as_table(rows, start)
    except Exception as e:
        return f"Error: {e}"


@tool
def find_rows(worksheet: str, query: str) -> str:
    """Find rows in a tab whose cells contain `query` (case-insensitive).
    Returns the header row plus each matching row with its row number."""
    try:
        rows = _ws(worksheet).get_all_values()
        if not rows:
            return "(empty)"
        q = query.lower()
        hits = [
            f"{i}\t" + "\t".join(r)
            for i, r in enumerate(rows, 1)
            if i > 1 and any(q in c.lower() for c in r)
        ]
        header = "1\t" + "\t".join(rows[0])
        return (
            header
            + "\n"
            + ("\n".join(hits[:MAX_ROWS_RETURNED]) if hits else "(no matches)")
        )
    except Exception as e:
        return f"Error: {e}"


@tool
def append_rows(worksheet: str, rows: list[list[str]]) -> str:
    """Append rows to the bottom of a tab. `rows` is a list of rows, each a list of
    cell values in column order matching the header. Values are parsed like typed
    input, so numbers, dates and formulas ('=SUM(B2:B9)') work."""
    try:
        res = _ws(worksheet).append_rows(rows, value_input_option="USER_ENTERED")
        return f"Appended {len(rows)} row(s): {res.get('updates', {}).get('updatedRange', '')}"
    except Exception as e:
        return f"Error: {e}"


@tool
def update_range(worksheet: str, a1_range: str, values: list[list[str]]) -> str:
    """Overwrite cells in a tab. a1_range is the top-left cell or full range in A1
    notation (e.g. 'C5' or 'B2:D3'); `values` is a 2D list of rows."""
    try:
        res = _ws(worksheet).update(values, a1_range, value_input_option="USER_ENTERED")
        return f"Updated {res.get('updatedCells', 0)} cell(s) in {res.get('updatedRange', a1_range)}"
    except Exception as e:
        return f"Error: {e}"


@tool
def add_worksheet(title: str, header: list[str] | None = None) -> str:
    """Create a new tab, optionally writing a header row."""
    try:
        ws = spreadsheet.add_worksheet(
            title=title, rows=1000, cols=max(26, len(header or []))
        )
        if header:
            ws.update([header], "A1")
        return f"Created tab {title!r}"
    except Exception as e:
        return f"Error: {e}"


WRITE_TOOLS = {"append_rows", "update_range", "add_worksheet"}

SYSTEM_PROMPT = """You are a spreadsheet assistant with access to one Google Sheets spreadsheet.

Rules:
- Start by calling list_worksheets to learn the tabs and their headers.
- Read before you write: check the header and existing rows so new values line up with the right columns.
- Read lines start with the sheet row number, which you can use to build A1 ranges for update_range.
- For large tabs, read narrower ranges or use find_rows instead of reading everything.
- Writes need human approval. If a write is rejected, don't retry it unchanged; ask what to do instead.
- Never claim a change was made unless the tool result confirms it.
"""

agent = create_deep_agent(
    model=model,
    name="Sheets_Agent",
    tools=[
        list_worksheets,
        read_range,
        find_rows,
        append_rows,
        update_range,
        add_worksheet,
    ],
    system_prompt=SYSTEM_PROMPT,
    interrupt_on={name: True for name in WRITE_TOOLS},
    checkpointer=MemorySaver(),
)


# ── Chat loop with approval for writes ───────────────────────────────────────


def _review(action_requests: list[dict]) -> list[dict]:
    decisions = []
    for req in action_requests:
        print(f"\n⚠  The agent wants to call {req['name']}:")
        print(json.dumps(req["args"], indent=2, ensure_ascii=False))
        choice = input("Approve? (y = yes / n = reject): ").strip().lower()
        if choice in ("y", "yes", "approve"):
            decisions.append({"type": "approve"})
        else:
            reason = input("Why? (optional, sent to the agent): ").strip()
            decisions.append(
                {"type": "reject", "message": reason or "User rejected this change."}
            )
    return decisions


def ask(question: str, config: dict) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        version="v2",
    )
    while result.interrupts:
        decisions = _review(result.interrupts[0].value["action_requests"])
        result = agent.invoke(
            Command(resume={"decisions": decisions}), config=config, version="v2"
        )
    return result.value["messages"][-1].text


def main() -> None:
    config = {"configurable": {"thread_id": f"sheets-{uuid.uuid4()}"}}
    print(f"Connected to spreadsheet: {spreadsheet.title}")

    if len(sys.argv) > 1:  # one-shot mode
        print(ask(" ".join(sys.argv[1:]), config))
        return

    print("Ask about your sheet (empty line or Ctrl-C to quit).")
    try:
        while question := input("\nYou: ").strip():
            print(f"\nAgent: {ask(question, config)}")
    except (KeyboardInterrupt, EOFError):
        pass


if __name__ == "__main__":
    main()

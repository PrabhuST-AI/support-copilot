"""agent_tools.py -- Tool specifications and implementations for the ReAct refund agent.

Tasks (4 marks):
Implement five tools with:
- Name, one-sentence description, JSON-style parameter specification in TOOLS_SPEC list.
- TOOL_MAP dictionary from tool name string to Python function.

Tools:
1. get_order(order_id):
   Reads data/instamart.db and returns {"order_id", "status", "order_amount", "delivery_minutes"}
   or {"error": "order not found"}.
2. search_policy(query):
   Returns retrieve(query) output under the key "results".
3. calculate_refund(order_amount, delivery_minutes):
   Returns {"refund_inr": <amount>} following refund rules (> 45 min -> full refund, <= 45 min -> 0).
4. issue_refund(order_id, refund_inr) and escalate_to_human(order_id, refund_inr):
   Each append one JSON line to outputs/refund_ledger.jsonl (with timestamp and tool name)
   and return {"refund_inr": <amount>}.
"""

from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

from vectorize_and_retrieve import retrieve

DB_PATH = "data/instamart.db"
LEDGER_PATH = "outputs/refund_ledger.jsonl"


def get_order(order_id: str) -> Dict[str, Any]:
    """Reads data/instamart.db and retrieves order details or an error dictionary."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
        (order_id,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}

    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]) if row[2] is not None else 0.0,
        "delivery_minutes": float(row[3]) if row[3] is not None else None,
    }


def search_policy(query: str) -> Dict[str, Any]:
    """Retrieves relevant policy passages using TF-IDF cosine similarity."""
    results = retrieve(query, k=3)
    return {"results": results}


def calculate_refund(order_amount: float, delivery_minutes: Optional[float]) -> Dict[str, float]:
    """Calculates refund amount based on delivery time threshold (> 45 minutes)."""
    if delivery_minutes is not None and float(delivery_minutes) > 45.0:
        return {"refund_inr": round(float(order_amount), 2)}
    return {"refund_inr": 0.0}


def _record_ledger_entry(tool_name: str, order_id: str, refund_inr: float) -> Dict[str, float]:
    """Appends an event record to outputs/refund_ledger.jsonl and returns refund amount."""
    os.makedirs(os.path.dirname(LEDGER_PATH), exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": str(order_id),
        "refund_inr": float(refund_inr),
    }
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": float(refund_inr)}


def issue_refund(order_id: str, refund_inr: float) -> Dict[str, float]:
    """Approves and issues an automated refund up to INR 1000, logging to ledger."""
    return _record_ledger_entry("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id: str, refund_inr: float) -> Dict[str, float]:
    """Escalates a refund exceeding INR 1000 to a human support lead, logging to ledger."""
    return _record_ledger_entry("escalate_to_human", order_id, refund_inr)


# =============================================================================
# TOOLS_SPEC: JSON-style parameter specifications
# =============================================================================
TOOLS_SPEC: List[Dict[str, Any]] = [
    {
        "name": "get_order",
        "description": "Fetch order details including status, order amount, and delivery minutes from the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique order identifier to look up (e.g. ORD-100048).",
                }
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Search the policy documents for relevant guidelines using semantic similarity retrieval.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query keywords regarding store policy.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Calculate eligible refund amount based on order spend and delivery duration under policy rules.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {
                    "type": "number",
                    "description": "Total order amount in INR.",
                },
                "delivery_minutes": {
                    "type": "number",
                    "description": "Total delivery time in minutes.",
                },
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Approve and issue an automated customer refund for amounts up to INR 1000, logging to ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier to be refunded.",
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The refund amount in INR to disburse.",
                },
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate a high-value refund request exceeding INR 1000 to a human support lead, logging to ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order identifier requiring manual review.",
                },
                "refund_inr": {
                    "type": "number",
                    "description": "The requested refund amount in INR requiring lead approval.",
                },
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

# =============================================================================
# TOOL_MAP: Name -> Callable
# =============================================================================
TOOL_MAP: Dict[str, Any] = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatches a tool call by name with keyword arguments via TOOL_MAP."""
    if tool_name not in TOOL_MAP:
        return {"error": f"Tool '{tool_name}' not found in TOOL_MAP"}
    func = TOOL_MAP[tool_name]
    return func(**arguments)


def main():
    print("=" * 80)
    print("  AGENT TOOLS AND TOOL MAP VERIFICATION")
    print("=" * 80)

    # 1. Verify TOOLS_SPEC structure
    assert len(TOOLS_SPEC) == 5, f"Expected 5 tool specs, found {len(TOOLS_SPEC)}"
    print(f"Verified TOOLS_SPEC contains {len(TOOLS_SPEC)} tool definitions:")
    for spec in TOOLS_SPEC:
        print(f"  - {spec['name']:<18}: {spec['description']}")

    # 2. Verify TOOL_MAP
    assert len(TOOL_MAP) == 5, f"Expected 5 tools in TOOL_MAP, found {len(TOOL_MAP)}"
    for name in TOOL_MAP:
        assert name in [s["name"] for s in TOOLS_SPEC], f"Mismatch for tool {name}"

    # 3. Test get_order
    res_ord_found = execute_tool("get_order", {"order_id": "ORD-100048"})
    print(f"\n[Test get_order (found)]: {res_ord_found}")
    assert res_ord_found["status"] == "completed"
    assert res_ord_found["order_amount"] == 128.25
    assert res_ord_found["delivery_minutes"] == 65.0

    res_ord_missing = execute_tool("get_order", {"order_id": "ORD-999999"})
    print(f"[Test get_order (not found)]: {res_ord_missing}")
    assert res_ord_missing == {"error": "order not found"}

    # 4. Test search_policy
    res_search = execute_tool("search_policy", {"query": "refund policy late delivery"})
    print(f"[Test search_policy]: Found {len(res_search['results'])} chunks (top source: {res_search['results'][0]['source']})")
    assert "results" in res_search and len(res_search["results"]) == 3

    # 5. Test calculate_refund
    res_calc_late = execute_tool("calculate_refund", {"order_amount": 128.25, "delivery_minutes": 65.0})
    print(f"[Test calculate_refund (>45m)]: {res_calc_late}")
    assert res_calc_late == {"refund_inr": 128.25}

    res_calc_ontime = execute_tool("calculate_refund", {"order_amount": 109.65, "delivery_minutes": 20.0})
    print(f"[Test calculate_refund (<=45m)]: {res_calc_ontime}")
    assert res_calc_ontime == {"refund_inr": 0.0}

    # 6. Test issue_refund & ledger recording
    res_issue = execute_tool("issue_refund", {"order_id": "ORD-100048", "refund_inr": 128.25})
    print(f"[Test issue_refund]: {res_issue}")
    assert res_issue == {"refund_inr": 128.25}

    # 7. Test escalate_to_human & ledger recording
    res_escalate = execute_tool("escalate_to_human", {"order_id": "ORD-100145", "refund_inr": 1999.00})
    print(f"[Test escalate_to_human]: {res_escalate}")
    assert res_escalate == {"refund_inr": 1999.00}

    # Verify ledger file exists and has entries
    assert os.path.exists(LEDGER_PATH), f"Ledger file {LEDGER_PATH} must exist"
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    print(f"\nLedger entries in {LEDGER_PATH}: {len(lines)} records logged.")
    assert len(lines) >= 2
    last_two = lines[-2:]
    assert last_two[0]["tool"] == "issue_refund" and last_two[0]["order_id"] == "ORD-100048"
    assert last_two[1]["tool"] == "escalate_to_human" and last_two[1]["order_id"] == "ORD-100145"

    print("\n" + "=" * 80)
    print("  ALL 5 TOOLS AND TOOL_MAP VERIFIED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()

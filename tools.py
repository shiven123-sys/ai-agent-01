

import ast
import json
import operator
import os
from datetime import datetime, timezone

import requests

NOTES_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "notes_storage", "notes.json")

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": f"Could not evaluate '{expression}': {e}"})

def get_current_datetime(timezone_offset_hours: float = 0) -> str:
    now = datetime.now(timezone.utc)
    if timezone_offset_hours:
        from datetime import timedelta
        now = now + timedelta(hours=timezone_offset_hours)
    return json.dumps({"utc_or_offset_time": now.strftime("%Y-%m-%d %H:%M:%S")}

def get_weather(city: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        ).json()
        results = geo.get("results")
        if not results:
            return json.dumps({"error": f"Could not find location '{city}'"})
        lat, lon = results[0]["latitude"], results[0]["longitude"]
        resolved_name = results[0].get("name", city)

        weather = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": "true"},
            timeout=10,
        ).json()
        current = weather.get("current_weather", {})
        return json.dumps({
            "location": resolved_name,
            "temperature_c": current.get("temperature"),
            "windspeed_kmh": current.get("windspeed"),
        })
    except Exception as e:
        return json.dumps({"error": f"Weather lookup failed: {e}"})

def search_wikipedia(query: str) -> str:
    try:
        resp = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(query)}",
            timeout=10,
            headers={"User-Agent": "agentic-ai-assistant-demo/1.0"},
        )
        if resp.status_code != 200:
            return json.dumps({"error": f"No Wikipedia page found for '{query}'"})
        data = resp.json()
        return json.dumps({
            "title": data.get("title"),
            "summary": data.get("extract", "")[:800],
        })
    except Exception as e:
        return json.dumps({"error": f"Wikipedia lookup failed: {e}"})

def _load_notes():
    if not os.path.exists(NOTES_PATH):
        return []
    with open(NOTES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_notes(notes):
    os.makedirs(os.path.dirname(NOTES_PATH), exist_ok=True)
    with open(NOTES_PATH, "w", encoding="utf-8") as f:
        json.dump(notes, f, indent=2)


def add_note(text: str) -> str:
    notes = _load_notes()
    entry = {"id": len(notes) + 1, "text": text, "created_at": datetime.now(timezone.utc).isoformat()}
    notes.append(entry)
    _save_notes(notes)
    return json.dumps({"status": "saved", "note": entry})


def list_notes(_: str = "") -> str:
    notes = _load_notes()
    if not notes:
        return json.dumps({"notes": [], "message": "No notes saved yet."})
    return json.dumps({"notes": notes})

_CONVERSIONS = {
    ("km", "miles"): lambda x: x * 0.621371,
    ("miles", "km"): lambda x: x / 0.621371,
    ("kg", "lb"): lambda x: x * 2.20462,
    ("lb", "kg"): lambda x: x / 2.20462,
    ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
    ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
    ("meters", "feet"): lambda x: x * 3.28084,
    ("feet", "meters"): lambda x: x / 3.28084,
}


def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    key = (from_unit.lower().strip(), to_unit.lower().strip())
    if key not in _CONVERSIONS:
        supported = sorted(set(a for a, _ in _CONVERSIONS)) + sorted(set(b for _, b in _CONVERSIONS))
        return json.dumps({"error": f"Unsupported conversion {key}. Supported units: {sorted(set(supported))}"})
    result = _CONVERSIONS[key](value)
    return json.dumps({"value": value, "from": from_unit, "to": to_unit, "result": round(result, 4)})


TOOL_REGISTRY = {
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "get_weather": get_weather,
    "search_wikipedia": search_wikipedia,
    "add_note": add_note,
    "list_notes": list_notes,
    "convert_units": convert_units,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a arithmetic math expression, e.g. '12 * (3 + 4) / 2'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A math expression to evaluate."}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": "Get the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_offset_hours": {
                        "type": "number",
                        "description": "Hours offset from UTC, e.g. 5.5 for IST. Default 0 (UTC).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name, e.g. 'Pune' or 'London'."}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_wikipedia",
            "description": "Get a short Wikipedia summary for a topic or person.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Topic to look up."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_note",
            "description": "Save a short note/reminder for the user to persistent storage.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string", "description": "The note content."}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List all previously saved notes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_units",
            "description": "Convert a value between units (km/miles, kg/lb, celsius/fahrenheit, meters/feet).",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number"},
                    "from_unit": {"type": "string"},
                    "to_unit": {"type": "string"},
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]


def execute_tool(name: str, arguments: dict) -> str:
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool '{name}'"})
    try:
        return fn(**arguments)
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {e}"})

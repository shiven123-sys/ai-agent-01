import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from agent.tools import calculator, convert_units, add_note, list_notes, execute_tool  # noqa: E402
from agent import config  # noqa: E402


def test_calculator_basic():
    result = json.loads(calculator("2 + 2"))
    assert result["result"] == 4


def test_calculator_order_of_operations():
    result = json.loads(calculator("12 * (3 + 4) / 2"))
    assert result["result"] == 42.0


def test_calculator_blocks_code_injection():
    result = json.loads(calculator('__import__("os").system("echo hacked")'))
    assert "error" in result


def test_convert_units_km_to_miles():
    result = json.loads(convert_units(10, "km", "miles"))
    assert abs(result["result"] - 6.2137) < 0.01


def test_convert_units_unsupported():
    result = json.loads(convert_units(10, "km", "banana"))
    assert "error" in result


def test_notes_roundtrip(tmp_path, monkeypatch):
    import agent.tools as tools_module
    monkeypatch.setattr(tools_module, "NOTES_PATH", str(tmp_path / "notes.json"))
    result = json.loads(add_note("Test note"))
    assert result["status"] == "saved"
    listed = json.loads(list_notes())
    assert len(listed["notes"]) == 1
    assert listed["notes"][0]["text"] == "Test note"


def test_execute_tool_unknown():
    result = json.loads(execute_tool("not_a_real_tool", {}))
    assert "error" in result


def test_config_reports_unconfigured_cleanly(monkeypatch):
    monkeypatch.setattr(config, "API_KEY", "")
    assert config.is_configured() is False
    assert "GROQ_API_KEY" in config.setup_instructions() or "OPENAI_API_KEY" in config.setup_instructions()

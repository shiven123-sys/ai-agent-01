

from openai import OpenAI

from . import config
from .tools import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools:
calculator, get_current_datetime, get_weather, search_wikipedia, add_note,
list_notes, and convert_units.

Use tools whenever they would give a more accurate or up-to-date answer than
your own knowledge (e.g. current weather, current time, math, saved notes).
Don't call a tool if you don't need one — for general conversation just reply
directly. Keep answers concise and friendly."""


class Agent:
    def __init__(self):
        if not config.is_configured():
            raise RuntimeError(config.setup_instructions())
        self.client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self):
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _trim_history(self):
        # Keep system prompt + last N messages so context doesn't grow unbounded
        if len(self.history) > config.MAX_HISTORY_MESSAGES + 1:
            self.history = [self.history[0]] + self.history[-config.MAX_HISTORY_MESSAGES:]

    def chat(self, user_message: str) -> dict:
        """
        Send a user message, run the tool loop, return the final answer.
        Returns: {"reply": str, "tool_calls": [ {name, arguments, result}, ... ]}
        """
        self.history.append({"role": "user", "content": user_message})
        tool_trace = []

        for _ in range(config.MAX_TOOL_ITERATIONS):
            response = self.client.chat.completions.create(
                model=config.MODEL,
                messages=self.history,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.4,
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                # Record the assistant's tool-call request in history
                self.history.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ],
                })

                for tc in msg.tool_calls:
                    import json
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}
                    result = execute_tool(tc.function.name, args)
                    tool_trace.append({"name": tc.function.name, "arguments": args, "result": result})
                    self.history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                
                continue

            
            self.history.append({"role": "assistant", "content": msg.content})
            self._trim_history()
            return {"reply": msg.content, "tool_calls": tool_trace}

        
        fallback = "I wasn't able to finish that request within the tool-call limit."
        self.history.append({"role": "assistant", "content": fallback})
        self._trim_history()
        return {"reply": fallback, "tool_calls": tool_trace}

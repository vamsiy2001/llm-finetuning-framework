"""
Customer Support Agent — wraps the fine-tuned model with tool-calling capability.

The agent can:
  - Look up order status
  - Initiate refunds / cancellations / returns
  - Unlock accounts and manage subscriptions
  - Fall back to the LLM for general support questions

Usage:
    python agent/support_agent.py --model_id vamsiyvk/customer-support-lora-r16
"""

import argparse
import inspect
import json
import os
from typing import Callable

import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent.tools import ALL_TOOLS

load_dotenv()
console = Console()

SYSTEM_PROMPT = """You are a helpful, professional customer support agent with access to real-time tools.

When a customer asks about a specific order, account, or requests an action (refund, cancellation, return, etc.),
use the appropriate tool to get accurate information before responding.

For general questions (shipping policies, return windows, etc.), answer directly.

Available tools: {tool_list}

To call a tool, respond with:
TOOL_CALL: <tool_name>({{"param": "value", ...}})

After getting the tool result, respond naturally to the customer."""


def _build_tool_schemas(tools: list[Callable]) -> list[dict]:
    """Build JSON schema descriptions from function signatures and docstrings."""
    schemas = []
    for fn in tools:
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""

        params = {}
        for name, param in sig.parameters.items():
            annotation = param.annotation
            type_str = "string"
            if annotation == float:
                type_str = "number"
            elif annotation == int:
                type_str = "integer"
            params[name] = {"type": type_str}

        schemas.append({
            "name": fn.__name__,
            "description": doc.split("\n")[0],
            "parameters": params,
        })
    return schemas


def _execute_tool(tool_name: str, tool_args: dict, tools: list[Callable]) -> str:
    """Find and execute the named tool with the given args."""
    tool_map = {fn.__name__: fn for fn in tools}
    if tool_name not in tool_map:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = tool_map[tool_name](**tool_args)
        return json.dumps(result, indent=2)
    except TypeError as e:
        return json.dumps({"error": f"Tool call failed: {e}"})


def _parse_tool_call(text: str) -> tuple[str | None, dict | None]:
    """Extract tool name and args from a TOOL_CALL: line, or return (None, None)."""
    for line in text.splitlines():
        if line.strip().startswith("TOOL_CALL:"):
            call_str = line.split("TOOL_CALL:", 1)[1].strip()
            paren_idx = call_str.index("(")
            tool_name = call_str[:paren_idx].strip()
            args_str = call_str[paren_idx + 1:call_str.rindex(")")].strip()
            try:
                tool_args = json.loads(args_str)
                return tool_name, tool_args
            except json.JSONDecodeError:
                return tool_name, {}
    return None, None


class SupportAgent:
    def __init__(self, model_id: str, tools: list[Callable] = None):
        self.tools = tools or ALL_TOOLS
        self.tool_schemas = _build_tool_schemas(self.tools)
        self.tool_list = ", ".join(s["name"] for s in self.tool_schemas)
        self.conversation_history = []

        device_str = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.device = device_str

        console.print(f"Loading [cyan]{model_id}[/cyan] on [yellow]{device_str}[/yellow]...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.float32 if device_str == "cpu" else torch.float16
        if device_str == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            self.model = self.model.to(device_str)
        self.model.eval()
        console.print("Agent ready.\n")

    def _generate(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            ) + "\nASSISTANT:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def chat(self, user_message: str) -> str:
        system = SYSTEM_PROMPT.format(tool_list=self.tool_list)

        messages = [{"role": "system", "content": system}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_message})

        response = self._generate(messages)

        # check if the model wants to call a tool
        tool_name, tool_args = _parse_tool_call(response)
        if tool_name:
            console.print(f"  [dim]→ calling tool: {tool_name}({tool_args})[/dim]")
            tool_result = _execute_tool(tool_name, tool_args or {}, self.tools)

            # feed the tool result back and regenerate
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Tool result:\n{tool_result}\n\nPlease respond to the customer based on this information.",
            })
            response = self._generate(messages)

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response

    def reset(self):
        self.conversation_history = []


def run_interactive(model_id: str):
    agent = SupportAgent(model_id)

    console.print(Panel(
        "Customer Support Agent\nType your question and press Enter. Type 'quit' to exit, 'reset' to start new conversation.",
        border_style="blue",
    ))

    while True:
        try:
            user_input = input("\nCustomer: ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            console.print("[dim]Conversation reset.[/dim]")
            continue

        response = agent.chat(user_input)
        console.print(f"\n[green]Agent:[/green] {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer support agent with tool use")
    parser.add_argument(
        "--model_id",
        default=os.getenv("FINETUNED_MODEL_ID", "vamsiyvk/customer-support-lora-r16"),
    )
    args = parser.parse_args()
    run_interactive(args.model_id)

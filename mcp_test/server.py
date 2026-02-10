import argparse
from dotenv import load_dotenv
from typing import Annotated

from pydantic import Field
from fastmcp import FastMCP


load_dotenv()
server = FastMCP("helpful_tools")


@server.tool(
    description="""Простой калькулятор для выполнения
             математических вычислений. Принимает строку с выражением
             (например, '5 * (10 + 2)') и возвращает результат.
             Поддерживаемые операции: +, -, *, /, //, math.sqrt"""
)
def calculator(
    expression: Annotated[
        str, Field(description="Математическое выражение для вычисления")
    ],
) -> str:
    try:
        # Using a safer eval
        allowed_chars = "0123456789+-*/(). "
        if all(char in allowed_chars for char in expression):
            return str(eval(expression))
        else:
            return "Error: Invalid characters in expression."
    except Exception as e:
        return f"Error: {e}"


@server.tool(
    description="""Данный инструмент позволяет узнать
                   актуальную погоду в Москве."""
)
def weather(expression: Annotated[str, Field(description="время суток")]) -> str:
    try:
        return "The weather is GOOD!"
    except Exception as e:
        return f"Error: {e}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", type=str, required=False, default="sse")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.transport == "stdio":
        server.run()
    else:
        server.run(
            transport="sse",
            host="127.0.0.1",
            port=8002,
            path="/mcp",
            log_level="debug",
        )

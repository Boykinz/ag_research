from dataclasses import dataclass
import asyncio

from autogen_core import (AgentId,
                          MessageContext,
                          RoutedAgent,
                          SingleThreadedAgentRuntime,
                          message_handler)


@dataclass
class TextMessage:
    content: str
    source: str


@dataclass
class ImageMessage:
    url: str
    source: str


class MyAgent(RoutedAgent):

    @message_handler
    async def on_text_message(self, message: TextMessage, ctx: MessageContext) -> None:
        print(f"Hello, {message.source}, you said {message.content}!")

    @message_handler
    async def on_image_message(self, message: ImageMessage, ctx: MessageContext) -> None:
        print(f"Hello, {message.source}, you sent me {message.url}!")


async def main():
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent("My Agent"))
    agent_id = AgentId("my_agent", "default")

    runtime.start()
    await runtime.send_message(TextMessage(content="Hello, World!", source="User"), agent_id)
    await runtime.send_message(ImageMessage(url="https://example.com/image.jpg", source="User"), agent_id)
    await runtime.stop_when_idle()


if __name__ == '__main__':
    asyncio.run(main())

from dataclasses import dataclass
import asyncio
import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error() 
from autogen_core import (AgentId,
                          MessageContext,
                          RoutedAgent,
                          message_handler,
                          SingleThreadedAgentRuntime)
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

from phi_inference import LLMConfig, LLM


@dataclass
class MyMessageType:
    content: str


class MyAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")


class MyAssistant(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        model_client = OpenAIChatCompletionClient(model="gpt-4o")
        self._delegate = AssistantAgent(name, model_client=model_client)

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        response = await self._delegate.on_messages(
            [TextMessage(content=message.content, source="user")], ctx.cancellation_token
        )
        print(f"{self.id.type} responded: {response.chat_message}")


class MyPhiAssistant(RoutedAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.llm_config = LLMConfig()
        self.llm = LLM(self.llm_config)

    @message_handler
    async def handle_my_message_type(self, message: MyMessageType, ctx: MessageContext) -> None:
        print(f"{self.id.type} received message: {message.content}")
        # response = self.llm(message.content)
        response = self.llm.__chat_call__(message.content)
        print(f"{self.id.type} responded: {response}")


async def main():
    runtime = SingleThreadedAgentRuntime()
    await MyAgent.register(runtime, "my_agent", lambda: MyAgent())
    # await MyAssistant.register(runtime, "my_assistant", lambda: MyAssistant("my_assistant"))
    await MyPhiAssistant.register(runtime, "my_assistant", lambda: MyPhiAssistant("my_assistant"))

    runtime.start()
    # await runtime.send_message(MyMessageType("Hello, World!"), AgentId("my_agent", "default"))

    content = "Hello!"
    while content != 'stop':
        await runtime.send_message(MyMessageType(content), AgentId("my_assistant", "default"))
        content = input('user: ')
    await runtime.stop()


if __name__ == '__main__':
    asyncio.run(main())

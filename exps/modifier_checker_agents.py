from dataclasses import dataclass
from typing import Callable
import asyncio

from autogen_core import (DefaultTopicId,
                          MessageContext,
                          RoutedAgent,
                          default_subscription,
                          message_handler,
                          AgentId,
                          SingleThreadedAgentRuntime)


@dataclass
class Message:
    content: int


@default_subscription
class ModifierAgent(RoutedAgent):
    '''Агент для изменения заданного числа'''

    def __init__(self, modify_val: Callable[[int], int]) -> None:
        super().__init__("A modifier agent.")
        self._modify_val = modify_val

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        val = self._modify_val(message.content)
        print(f"{'-'*80}\nModifier:\nModified {message.content} to {val}")
        await self.publish_message(Message(content=val), DefaultTopicId())


@default_subscription
class CheckerAgent(RoutedAgent):
    '''Агент для проверки заданного числа на соответствие условию'''

    def __init__(self, run_until: Callable[[int], bool]) -> None:
        super().__init__("A checker agent.")
        self._run_until = run_until

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        if not self._run_until(message.content):
            print(f"{'-'*80}\nChecker:\n{message.content} passed the check, continue.")
            await self.publish_message(Message(content=message.content), DefaultTopicId())
        else:
            print(f"{'-'*80}\nChecker:\n{message.content} failed the check, stopping.")


async def main(value=10):
    runtime = SingleThreadedAgentRuntime()

    await ModifierAgent.register(
        runtime,
        "modifier",
        # изменение числа путём вычитания 1
        lambda: ModifierAgent(modify_val=lambda x: x - 1)
    )

    await CheckerAgent.register(
        runtime,
        "checker",
        # проверка числа до тех пор пока значение > 1
        lambda: CheckerAgent(run_until=lambda x: x <= 1)
    )

    # старт runtime и отправка сообщения агенту CheckerAgent
    runtime.start()
    message = Message(content=value)
    await runtime.send_message(message, AgentId("checker", "default"))
    await runtime.stop_when_idle()


if __name__ == '__main__':
    asyncio.run(main())

# попробовать qwen3-32b

from dataclasses import dataclass
import logging
from typing import Any, List, Union
import asyncio
import json
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler # Для ротации логов, если файл станет большим
import sys # Для управления обработчиками логгера

from autogen_core import (
    SingleThreadedAgentRuntime,
    TypeSubscription,
    message_handler,
    RoutedAgent,
    MessageContext,
    TopicId,
    ClosureContext,
    ClosureAgent
)
from autogen_core.models import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionExecutionResultMessage,
    FunctionExecutionResult
)
from autogen_core.models import UserMessage, AssistantMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import McpWorkbench, SseServerParams, mcp_server_tools


# Базовая настройка логгирования на уровне WARNING для консоли (можно изменить при необходимости)
logging.basicConfig(level=logging.WARNING, handlers=[logging.StreamHandler(sys.stdout)])

# Создадим логгер для текущего модуля, но не будем настраивать его здесь
logger = logging.getLogger(__name__)


# Функция для настройки логгера на запись в файл с заданным именем
def setup_logger_to_file(filename: str, logger_name: str = __name__, log_level: int = logging.INFO):
    """Настраивает логгер для записи в указанный файл."""
    # Получаем логгер
    specific_logger = logging.getLogger(logger_name)
    specific_logger.setLevel(log_level)

    # Удаляем существующие обработчики, чтобы избежать дублирования при повторных вызовах
    for handler in specific_logger.handlers[:]:
        specific_logger.removeHandler(handler)
        handler.close() # Закрываем обработчики

    # Создаем обработчик для записи в файл
    # RotatingFileHandler поможет, если логи будут большими
    #file_handler = RotatingFileHandler(filename, maxBytes=10*1024*1024, backupCount=5) # 10MB, 5 архивных копий
    file_handler = RotatingFileHandler(filename, maxBytes=10*1024*1024, encoding="utf-8")
    file_handler.setLevel(log_level)

    # Создаем форматтер и добавляем его к обработчику
    formatter = logging.Formatter(
        '\n%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
    )
    file_handler.setFormatter(formatter)

    # Добавляем обработчик к логгеру
    specific_logger.addHandler(file_handler)

    # Отключаем propagate, чтобы логи не шли в родительские логгеры (например, root)
    specific_logger.propagate = False

    return specific_logger


@dataclass
class InMessage:
    query: str
    evaluate_flag: bool


@dataclass
class ResultMessage:
    query: str
    answer: str


@dataclass
class MetricMessage(ResultMessage):
    eval_answer: Any


RESULT_ATTRIBUTE_QUEUE = asyncio.Queue[ResultMessage | MetricMessage]()


async def output_result(
    _agent: ClosureContext, message: ResultMessage | MetricMessage, ctx: MessageContext
) -> None:
    await RESULT_ATTRIBUTE_QUEUE.put(message)


class ReActAgent(RoutedAgent):
    def __init__(self, name: str, model_client: OpenAIChatCompletionClient, server_params, logger_instance):
        super().__init__(name)
        self.__model_client = model_client
        self.server_params = server_params
        self.logger = logger_instance # Сохраняем экземпляр логгера

        # путь до промпта вынести в параметры хмммм
        
        #self.system_prompt = """
        #Ты — интеллектуальный ассистент, задача которого — решать поставленные пользователем задачи посредством пошагового вызова инструментов. Твоя основная цель — минимизировать неопределённость, анализируя доступную информацию, и принимать обоснованные решения на каждом этапе.
        
        #Дополнительная информация:
        #1. Информация о показателях эффективности доступна при использовании рага по методологии
        #2. Дополнительную информациию о конкреных ДО можно найти в базе данных с помощью инструмента, преобразующего текст в sql запросы
        #"""
        self.system_prompt = """
        Ты — интеллектуальный ассистент, задача которого — решать поставленные пользователем задачи посредством пошагового вызова инструментов. Твоя основная цель — минимизировать неопределённость, анализируя доступную информацию, и принимать обоснованные решения на каждом этапе.
        """
        self.critic_step_prompt = "Если есть инструменты, которые потенциально могут снять неопределенность и/или помочь решить поставленную задачу, то вызови их, в противном случае сформируй финальный ответ."
        
        self.history = []

    @message_handler
    async def get_result_message(self, message: InMessage, ctx: MessageContext) -> None:
        query = message.query
        evaluate_flag = message.evaluate_flag
        self.logger.info(f"Запроса пользователя:\n{query}")
        self.logger.info(f"Системный промпт:\n{self.system_prompt}")
        self.logger.info(f"Запрос для доп шага (может быть как финальным, так и промежуточным):\n{self.critic_step_prompt}")

        async with McpWorkbench(server_params=self.server_params) as workbench:
            self.history = [SystemMessage(content=self.system_prompt)] + [UserMessage(content=query, source="user")]
            wb_tools = await workbench.list_tools()
            wb_tools_str = '\n'.join([t['name'] + ":\n\n\n" + t['description'] + "\n\n\n" for t in wb_tools])
            self.logger.info(f"Доступные инструменты:\n{wb_tools_str}")
            #create_result = await self.__model_client.create(
            #    messages=self.history,
            #    tools=wb_tools,  # если у вас нет инструментов, передайте пустой список
            #    cancellation_token=ctx.cancellation_token,
            #)
            create_result = None
            stream = self.__model_client.create_stream(
                messages=self.history,
                tools=wb_tools,  # если у вас нет инструментов, передайте пустой список
                cancellation_token=ctx.cancellation_token,
                )
            self.logger.info("Запрос к модели (шаг 1)")
            streamed_content = ""
            #print("Streamed responses:")
            async for response in stream:
                if isinstance(response, str):
                    # A partial response is a string.
                    #print(response, flush=True, end="")
                    streamed_content += response
                else:
                    # The last response is a CreateResult object with the complete message.
                    #print("\n\n------------\n")
                    #print("The complete response:", flush=True)
                    #print(response.content, flush=True)
                    create_result = response
                    
                    if streamed_content:
                        self.logger.info(f"Ответ модели (stream) (шаг 1):\n{streamed_content}")
                    self.logger.info(f"Ответ модели (шаг 1):\n{response.content}")
            
            # если есть вызовы инструментов
            
            iteration = 1
            while isinstance(create_result.content, list):
                iteration += 1
                create_result.content = [create_result.content[0]]
                self.history.append(AssistantMessage(content=create_result.content, source="assistant"))
                for item in create_result.content:
                    self.logger.info(f"Вызов инструмента: {item.name}")
                    self.logger.info(f"Аргументы инструмента {item.name}: {item.arguments}")
                    
                    try:
                        tool_result = await workbench.call_tool(
                            item.name,
                            json.loads(item.arguments)
                        )
                        # Предполагаем, что результат содержит список с одним элементом content
                        tool_output_content = tool_result.result[0].content if tool_result.result else "Нет результата"
                        self.logger.info(f"Результат работы инструмента {item.name}:\n{tool_output_content}")
                        exec_result = FunctionExecutionResult(
                            call_id=item.id,  # type: ignore
                            content=tool_result.result[0].content,
                            is_error=False,
                            name=item.name,
                        )
                        self.history.append(
                            FunctionExecutionResultMessage(content=[exec_result])
                        )
                    except Exception as e:
                        self.logger.error(f"Ошибка при вызове инструмента {item.name}:\n{e}")
                        error_result = FunctionExecutionResult(
                            call_id=item.id,
                            content=f"Ошибка вызова инструмента: {str(e)}",
                            is_error=True,
                            name=item.name,
                        )
                        self.history.append(
                            FunctionExecutionResultMessage(content=[error_result])
                        )
                #create_result = await self.__model_client.create(
                #    messages=self.history,
                #    tools=wb_tools,  # если у вас нет инструментов, передайте пустой список
                #    cancellation_token=ctx.cancellation_token,
                #)
                stream = self.__model_client.create_stream(
                    messages=self.history,
                    tools=wb_tools,  # если у вас нет инструментов, передайте пустой список
                    cancellation_token=ctx.cancellation_token,
                )
                self.logger.info(f"Запрос к модели (шаг {iteration}):")
                streamed_content = ""
                #print("Streamed responses:")
                async for response in stream:
                    if isinstance(response, str):
                        # A partial response is a string.
                        #print(response, flush=True, end="")
                        streamed_content += response
                    else:
                        # The last response is a CreateResult object with the complete message.
                        #print("\n\n------------\n")
                        #print("The complete response:", flush=True)
                        #print(response.content, flush=True)
                        create_result = response
                        if streamed_content:
                            self.logger.info(f"Ответ модели (stream) (шаг {iteration}): {streamed_content}")
                        self.logger.info(f"Ответ модели (шаг {iteration}):\n{response.content}")
                if isinstance(create_result.content, list):
                    continue
                self.history.append(AssistantMessage(content=create_result.content, source="assistant"))
                self.history = self.history + [UserMessage(content=self.critic_step_prompt, source="user")]
                self.logger.info("Доп шаг (может быть как финальным, так и промежуточным)")
                iteration += 1
                #create_result = await self.__model_client.create(
                #    messages=self.history,
                #    tools=wb_tools,  # если у вас нет инструментов, передайте пустой список
                #    cancellation_token=ctx.cancellation_token,
                #)
                stream = self.__model_client.create_stream(
                    messages=self.history,
                    tools=wb_tools,  # если у вас нет инструментов, передайте пустой список
                    cancellation_token=ctx.cancellation_token,
                )
                self.logger.info(f"Запрос к модели (шаг {iteration}):")
                streamed_content = ""
                #print("Streamed responses:")
                async for response in stream:
                    if isinstance(response, str):
                        # A partial response is a string.
                        #print(response, flush=True, end="")
                        streamed_content += response
                    else:
                        # The last response is a CreateResult object with the complete message.
                        #print("\n\n------------\n")
                        
                        #print("The complete response:", flush=True)
                        #print(response.content, flush=True)
                        create_result = response
                        if streamed_content:
                            self.logger.info(f"Ответ модели (stream) (шаг {iteration}):\n{streamed_content}")
                        self.logger.info(f"Ответ модели (шаг {iteration}):\n{response.content}")
                
        # Отправка финального результата
        final_answer = create_result.content if create_result else "Не удалось получить ответ."
        self.logger.info(f"Финальный ответ для пользователя:\n{final_answer}")
        await self.publish_message(
            ResultMessage(query=query, answer=create_result.content),
            TopicId("result_topic", self.id.key),
        )


class ReActPipeline:
    def __init__(self, llm_host, model_name, server_params) -> None:
        self.__runtime = SingleThreadedAgentRuntime()
        
        self.llm_host = llm_host
        self.server_params =server_params
        
        self.__client = OpenAIChatCompletionClient(
            model=model_name,
            api_key="-",
            base_url=self.llm_host,
            model_info={
                "json_output": False,
                "function_calling": True,
                "vision": False,
                "family": "unknown",
                "structured_output": False,
            },
        )
        
    async def __register_agents(self, logger_instance):
        await ReActAgent.register(
            self.__runtime,
            "react_agent",
            lambda: ReActAgent("ReAct agent", self.__client, self.server_params, logger_instance),
        )
        await ClosureAgent.register_closure(
            self.__runtime,
            "output_result",
            output_result,
            subscriptions=lambda: [
                TypeSubscription(topic_type="result_topic", agent_type="output_result")
            ],
        )

    async def __add_subs(self):
        # сообщение от пользователя сначала прилетает интент рекогнишн агенту
        await self.__runtime.add_subscription(
            TypeSubscription(
                topic_type="user_query_topic", agent_type="react_agent"
            )
        )
        
    async def __run_pipeline(self, messages: list[InMessage], log_filename: str):
        final_result = None
        # Настройка логгера для этого конкретного запуска
        pipeline_logger = setup_logger_to_file(log_filename)
        await self.__register_agents(pipeline_logger) # Передаем логгер агенту
        await self.__add_subs()

        self.__runtime.start()

        for message in messages:
            await self.__runtime.publish_message(
                message,
                topic_id=TopicId("user_query_topic", "default"),
            )

        await self.__runtime.stop_when_idle()

        while not RESULT_ATTRIBUTE_QUEUE.empty():
            # final_result = (result := await RESULT_ATTRIBUTE_QUEUE.get()).answer
            final_result = await RESULT_ATTRIBUTE_QUEUE.get()

        return final_result

    def __call__(self, messages: list[InMessage], log_filename: str = "pipeline.log"):
        result = asyncio.run(self.__run_pipeline(messages, log_filename))

        return result

# **********************************************************

load_dotenv()
llm_host = os.getenv("LLM_URL")
mcp_server_url = os.getenv("MCP_SERVER_URL")
model_name = os.getenv("MODEL_NAME")

server_params = SseServerParams(url=mcp_server_url)

pipeline = ReActPipeline(llm_host, model_name, server_params)
#in_msg = "какие показатели эффективности посчитаны для до анапский"
#in_msg = "в методологии выбери любой показатель эффективности(используй инструмент раг по методологии) и найди для до 'анапский' его значение(используй инструмент text to sql)"
#in_msg = "в методологии выбери любой показатель эффективности и найди для до 'анапский' его значение"
#in_msg = "в методологии найди найди критические значения для среднего времени ожидания и проанализируй на основе данных из бд до анапский по данной метрике"

#in_msg = "выбери любой показатель эффективности из методологии и найди для до 'анапский' его значение"
#in_msg = "для показателя эффективности 'уровень нагрузки' узнай нормальное значение и проанализируй на сколько среднее значение этого показателя у ДО 'Кореновский' отличается от нормального значения"
#in_msg = "узнай критическое значение показателя эффективности процент сброшенных талонов и напиши количество наблюдений в бд, относящихся к ДО Белорененский, которые превышают данное значение"
#in_msg = "узнай критическое значение показателя эффективности процент сброшенных талонов и напиши количество наблюдений в бд, относящихся к ДО Белорененский, которые меньше этого значения"

in_msg_lst = [
    "принципы эффективной работы менеджмента",
    "Что такое ДО?",
    "Какие еще есть названия у ДО?",
    "как расшифровывается Управляющий РОО/БГ?",
    "1+1?",
    "Посчитай 4*13 + 3 - 7 - 9+123",
    "какие до имеют худшие результаты по доле обслуженных клиентов?",
    "какая была доля обслуженных клиентов у до анапский в 23 февраля?",
    "какие показатели эффективности посчитаны для до анапский",
    "выбери любой показатель эффективности из методологии и найди для до 'анапский' его значение",
    "для показателя эффективности 'уровень нагрузки' узнай нормальное значение и проанализируй на сколько среднее значение этого показателя у ДО 'Кореновский' отличается от нормального значения",
    "узнай критическое значение показателя эффективности процент сброшенных талонов и напиши количество наблюдений в бд, относящихся к ДО Белорененский, которые превышают данное значение",
    "Узнай, что такое ДО, и извлеки из бд 3 случайных наименования",
    "Для какой точки, в базе содержится наибольшее количество наблюдений?",
    "извлеки все значения среднего времени обслуживания для ДО Армавирский и посчитай среднее с помощью инструмента калькулятор",
    "извлеки все значения показателя уровень нагрузки для ДО Геленджик и сложи их",
    "ИСпользуя калбкулятор скажи, сколько всего человек обслужило ДО Ейский"
]    

#in_msg = "привет"

#message = InMessage(
#    in_msg,    
#    False,
#    )
# Вызов pipeline с указанием имени файла для логов
#log_file_name = f"pipeline_run_{int(asyncio.get_event_loop().time())}.log" # Пример динамического имени
#print(pipeline([message], log_filename=log_file_name))


for idx, in_msg in enumerate(in_msg_lst):
    pipeline = ReActPipeline(llm_host, model_name, server_params)    # у меня возникает ошибка без вот этой строчки, НАДО ПОФИКСИТЬ!!!!!!!!!!!
    log_file_name = f"./run_3/pipeline_run_{idx}.log"
    message = InMessage(
        in_msg,    
        False,
    )
    print(pipeline([message], log_filename=log_file_name))

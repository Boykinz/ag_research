from __future__ import annotations

import  torch
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer)
from transformers import pipeline

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from phi_inference import LLMConfig


class LLM:

    def __init__(self, cfg: LLMConfig):
        self.model_name = cfg.model_name
        self.device = cfg.device
        self.model = self.get_model()
        self.tokenizer = self.get_tokenizer()
        self.generator = self.get_generator(**cfg.pipeline_config)
        self.messages = [{
                "role": "system",
                "content": "You are helpfull AI-assistant."
            }]

    def get_model(self, torch_dtype='auto', trust_remote_code=True):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code
        )
        return model

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer

    def get_generator(self, **kwargs):
        generator = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            **kwargs)
        return generator

    def encode(self, prompt, return_list=True):
        input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids
        print(input_ids)
        return input_ids[0] if return_list else input_ids

    def decode(self, output_ids):
        output = self.tokenizer.decode(output_ids)
        print(output)
        return output

    def __call__(self, prompt, return_text=True):
        messages = [
            {'role': 'user', 'content': prompt}
        ]
        output = self.generator(messages)
        return output[0]["generated_text"] if return_text else output

    def __chat_call__(self, prompt, return_text=True):
        self.messages.append(
            {'role': 'user', 'content': prompt}
        )
        output = self.generator(self.messages)
        generated_text = output[0]["generated_text"]
        self.messages.append(
            {'role': 'assistant', 'content': generated_text}
        )
        return generated_text if return_text else output


class LLMConfig:
    def __init__(self):
        # self.model_name = 'microsoft/Phi-3-mini-4k-instruct'
        self.model_name = "microsoft/phi-4"
        self.pipeline_config = {
            'task': 'text-generation',
            'return_full_text': False,
            # 'max_new_tokens': 500,
            'max_new_tokens': 16_000,
            'do_sample': False,
            'torch_dtype': torch.bfloat16
        }
        self.device = 'cuda:0'


def main(prompt="find x: x^2 - 2*x - 8 = 0"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, device count: {torch.cuda.device_count()}')
    config = LLMConfig()
    llm = LLM(config)
    output = llm(prompt)
    print(f'output: {output}')


def chat_main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, device count: {torch.cuda.device_count()}')
    config = LLMConfig()
    llm = LLM(config)
    while True:
        output = llm.__chat_call__(input('user:'))
        print(f'output: {output}')


if __name__ == '__main__':
    # main()
    chat_main()

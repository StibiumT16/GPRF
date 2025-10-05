from typing import List
from copy import deepcopy
import numpy as np
from transformers import AutoTokenizer
from .utils import resolve_max_tokens

class BaseGenerator:
    """`BaseGenerator` is a base object of Generator model."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()
    
    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()
    def update_base_setting(self):
        self.model_name = self._config.generator_model
        self.model_path = self._config.generator_model_path

        self.max_input_len = self._config.generator_max_input_len
        self.batch_size = self._config.generator_batch_size
        self.device = self._config.device
        self.gpu_num = self._config.gpu_num
        self.generation_params = self._config.generation_params
    
    def update_additional_setting(self):
        pass

    def generate(self, input_list: list) -> List[str]:
        pass


class VLLMGenerator(BaseGenerator):
    """Class for decoder-only generator, based on vllm."""

    def __init__(self, config):
        super().__init__(config)
        
        from vllm import LLM
        if self.use_lora:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self._config.gpu_memory_utilization,
                enable_lora = True,
                max_lora_rank = 64,
                max_logprobs = 32016,
                max_model_len = self.max_model_len
            )
        else:
            self.model = LLM(
                self.model_path,
                tensor_parallel_size = self.tensor_parallel_size,
                gpu_memory_utilization = self._config.gpu_memory_utilization,
                max_logprobs = 32016,
                max_model_len = self.max_model_len
            )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
    def update_additional_setting(self):
        '''
        if "gpu_memory_utilization" not in self._config:
            self.gpu_memory_utilization = 0.85
        else:
            self.gpu_memory_utilization = self._config.gpu_memory_utilization
        '''
        if self.gpu_num != 1 and self.gpu_num % 2 != 0:
            self.tensor_parallel_size = self.gpu_num - 1
        else:
            self.tensor_parallel_size = self.gpu_num

        self.lora_path = self._config.generator_lora_path
        self.use_lora = False
        if self.lora_path is not None:
            self.use_lora = True
        self.max_model_len = (self._config.generator_max_input_len + self.generation_params['max_tokens']) * 1.5

    def generate(
        self,
        input_list: List[str],
        return_raw_output=False,
        return_scores=False,
        **params,
    ):
        from vllm import SamplingParams

        if isinstance(input_list, str):
            input_list = [input_list]

        generation_params = deepcopy(self.generation_params)
        generation_params.update(params)
        if "do_sample" in generation_params:
            do_sample_flag = generation_params.pop("do_sample")
            if not do_sample_flag:
                generation_params["temperature"] = 0
        generation_params["seed"] = self._config.seed

        # handle param conflict
        generation_params = resolve_max_tokens(params, generation_params, prioritize_new_tokens=False)

        # fix for llama3
        if "stop" in generation_params:
            generation_params["stop"].append("<|eot_id|>")
            generation_params["include_stop_str_in_output"] = True
        else:
            generation_params["stop"] = ["<|eot_id|>"]

        if return_scores:
            if "logprobs" not in generation_params:
                generation_params["logprobs"] = 100

        sampling_params = SamplingParams(**generation_params)

        if self.use_lora:
            from vllm.lora.request import LoRARequest

            outputs = self.model.generate(
                input_list,
                sampling_params,
                lora_request=LoRARequest("lora_module", 1, self.lora_path),
            )
        else:
            outputs = self.model.generate(input_list, sampling_params)

        if return_raw_output:
            base_output = outputs
        else:
            if "n" in generation_params and generation_params["n"] > 1:
                generated_texts = [[output.outputs[i].text for i in range(len(output.outputs))] for output in outputs]
                base_output = generated_texts
            else:
                generated_texts = [output.outputs[0].text for output in outputs]
                base_output = generated_texts
        if return_scores:
            scores = []
            for output in outputs:
                logprobs = output.outputs[0].logprobs
                scores.append([np.exp(list(score_dict.values())[0].logprob) for score_dict in logprobs])
            return base_output, scores
        else:
            return base_output


def get_generator(config):
    return VLLMGenerator(config)


class GeneratorConfig:
    def __init__(
        self,
        generator_model,
        generator_model_path,
        gpu_num,
        seed=42,
        device="cuda",
        generator_max_input_len=2048,
        generator_batch_size = 1,
        gpu_memory_utilization = 0.85,
        generator_lora_path=None,
        generation_params={'do_sample': False, 'max_tokens' : 128}
    ):
        self.generator_model = generator_model
        self.generator_model_path = generator_model_path

        self.generator_max_input_len = generator_max_input_len
        self.device = device
        self.gpu_num = gpu_num
        self.seed = seed
        self.generator_batch_size = generator_batch_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.generator_lora_path = generator_lora_path
        self.generation_params = generation_params
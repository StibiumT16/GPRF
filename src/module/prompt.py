from typing import List
from transformers import AutoTokenizer


class BasePromptTemplate:
    def __init__(self, config):
        self.config = config
        self.max_input_len = config.generator_max_input_len
        self.generator_path = config.generator_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.generator_path, trust_remote_code=True)
    
    def truncate_prompt(self, prompt):
        assert isinstance(prompt, str)
        tokenized_prompt = self.tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
        if len(tokenized_prompt) > self.max_input_len:
            print(f"The input text length is greater than the maximum length ({len(tokenized_prompt)} > {self.max_input_len}) and has been truncated!")
            half = int(self.max_input_len / 2)
            prompt = self.tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                    self.tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        return prompt

    def format_reference(self, retrieval_result):
        format_reference = ""
        for idx, content in enumerate(retrieval_result):
            format_reference += f"Passage {idx+1}: {content}\n"

        return format_reference


class HyDEPromptTemplate(BasePromptTemplate):
    # msmarco dl19 dl20 dbpedia hotpotqa nq
    search_system_prompt = "Please write a passage to answer the question."
    search_user_prompt = "Question: {question}\nPassage:"
    
    # fever
    fever_system_prompt = "Please write a passage to support/refute the claim."
    fever_user_prompt = "Claim: {question}\nPassage:"
    
    # scifact Climate-FEVER
    scifact_system_prompt = "Please write a scientific paper passage to support/refute the claim."
    scifact_user_prompt = "Claim: {question}\nPassage:"
    
    # ArguAna touche2020
    arguana_system_prompt = "Please write a counter argument for the passage."
    arguana_user_prompt = "Passage: {question}\nCounter Argument:"
    
    # covid NFCorpus
    covid_system_prompt = "Please write a scientific paper passage to answer the question."
    covid_user_prompt = "Question: {question}\nPassage:"
    
    # scidocs
    scidocs_system_prompt = "Please write a scientific paper passage that is relevant and likely to be cited by the given query paper."
    scidocs_user_prompt = "Query Paper: {question}\nPassage:"
    
    # fiqa
    fiqa_system_prompt = "Please write a financial article passage to answer the question."
    fiqa_user_prompt = "Question: {question}\nPassage:"
    
    # quora
    quora_system_prompt = "Please rewrite the following question without changing its meaning."
    quora_user_prompt = "Question: {question}\nParaphrased Question:"
    
    def __init__(self, task, config, enable_chat=True):

        super().__init__(config)

        if task.lower() in ['fever']:
            print(f"\n=== Task: {task}, Prompt: FEVER ===\n")
            self.system_prompt = self.fever_system_prompt
            self.user_prompt = self.fever_user_prompt
        elif task.lower() in ['scifact', 'climate-fever']:
            print(f"\n=== Task: {task}, Prompt: Scifact ===\n")
            self.system_prompt = self.scifact_system_prompt
            self.user_prompt = self.scifact_user_prompt
        elif task.lower() in ['arguana', 'touche2020']:
            print(f"\n=== Task: {task}, Prompt: Arguana ===\n")
            self.system_prompt = self.arguana_system_prompt
            self.user_prompt = self.arguana_user_prompt
        elif task.lower() in ['trec-covid', 'nfcorpus']:
            print(f"\n=== Task: {task}, Prompt: Covid ===\n")
            self.system_prompt = self.covid_system_prompt
            self.user_prompt = self.covid_user_prompt
        elif task.lower() in ['scidocs']:
            print(f"\n=== Task: {task}, Prompt: SCIDOCS ===\n")
            self.system_prompt = self.scidocs_system_prompt
            self.user_prompt = self.scidocs_user_prompt
        elif task.lower() in ['fiqa']:
            print(f"\n=== Task: {task}, Prompt: FiQA ===\n")
            self.system_prompt = self.fiqa_system_prompt
            self.user_prompt = self.fiqa_user_prompt
        elif task.lower() in ['quora']:
            print(f"\n=== Task: {task}, Prompt: Quora ===\n")
            self.system_prompt = self.quora_system_prompt
            self.user_prompt = self.quora_user_prompt
        else: # msmarco dl19 dl20 dbpedia hotpotqa nq
            print(f"\n=== Task: {task}, Prompt: Search ===\n")
            self.system_prompt = self.search_system_prompt
            self.user_prompt = self.search_user_prompt
            
        self.enable_chat = enable_chat
        
    def get_string(self, question : str, **params):

        input_params = {"question": question}
        input_params.update(**params)
        
        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            
            input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        return self.truncate_prompt(input)


class LameRPromptTemplate(BasePromptTemplate):
    # msmarco dl19 dl20 dbpedia hotpotqa nq fever
    search_system_prompt =  "Give a question '{question}' and its possible answering passages (most of these passages are wrong) enumerated as:\n{documents}"
    search_user_prompt = "Please write a correct answering passage."
    
    # scifact Climate-FEVER covid NFCorpus
    scifact_system_prompt = "Give a question '{question}' and its possible scientific paper passages (most of these passages are wrong) enumerated as:\n{documents}"
    scifact_user_prompt = "Please write a correct scientific paper passage."
    
    # ArguAna touche2020
    arguana_system_prompt = "Give a question '{question}' and its possible counter-argument passages (most of these passages are wrong) enumerated as:\n{documents}"
    arguana_user_prompt = "Please write a correct counter-argument passage."
    
    # covid NFCorpus
    # The same as scifact
    
    #scidocs
    scidocs_system_prompt = "Give a scientific passage '{question}' and several passages it may cite (most of these passages are wrong) enumerated as:\n{documents}"
    scidocs_user_prompt = "Please write a correct citation passage."
    
    # fiqa
    fiqa_system_prompt = "Give a question '{question}' and its possible answering financial article passages (most of these passages are wrong) enumerated as:\n{documents}"
    fiqa_user_prompt = "Please write a correct answering financial article passage."
    
    # quora TODO
    quora_system_prompt = "Give a question '{question}' and its possible synonymous questions (most of these questions are wrong) enumerated as:\n{documents}"
    quora_user_prompt = "Please write a correct synonymous question."
    
    
    def __init__(self, task, config, enable_chat=True):

        super().__init__(config)

        if task.lower() in ['scifact', 'climate-fever', 'trec-covid', 'nfcorpus']:
            print(f"\n=== Task: {task}, Prompt: Scifact ===\n")
            self.system_prompt = self.scifact_system_prompt
            self.user_prompt = self.scifact_user_prompt
        elif task.lower() in ['arguana', 'touche2020']:
            print(f"\n=== Task: {task}, Prompt: Arguana ===\n")
            self.system_prompt = self.arguana_system_prompt
            self.user_prompt = self.arguana_user_prompt
        elif task.lower() in ['scidocs']:
            print(f"\n=== Task: {task}, Prompt: SCIDOCS ===\n")
            self.system_prompt = self.scidocs_system_prompt
            self.user_prompt = self.scidocs_user_prompt
        elif task.lower() in ['fiqa']:
            print(f"\n=== Task: {task}, Prompt: FiQA ===\n")
            self.system_prompt = self.fiqa_system_prompt
            self.user_prompt = self.fiqa_user_prompt
        elif task.lower() in ['quora']:
            print(f"\n=== Task: {task}, Prompt: Quora ===\n")
            self.system_prompt = self.quora_system_prompt
            self.user_prompt = self.quora_user_prompt
        else: # msmarco dl19 dl20 dbpedia hotpotqa nq fever
            print(f"\n=== Task: {task}, Prompt: Search ===\n")
            self.system_prompt = self.search_system_prompt
            self.user_prompt = self.search_user_prompt
            
        self.enable_chat = enable_chat
    
    def get_string(self, question : str, passages : List[str], **params):

        input_params = {"question": question, "documents": self.format_reference(passages)}
        input_params.update(**params)
        
        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            
            input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        return self.truncate_prompt(input)


class CoTPromptTemplate(BasePromptTemplate):
    base_system_prompt = ""
    base_user_prompt = "Answer the following query:\n\n{question}\nGive the rationale before answering:"
    
    def __init__(self, config, enable_chat=True):
        super().__init__(config)
        self.enable_chat = enable_chat
        
        self.system_prompt = self.base_system_prompt
        self.user_prompt = self.base_user_prompt
        
    def get_string(self, question : str, **params):
        input_params = {"question": question}
        input_params.update(**params)
        
        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            
            input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        return self.truncate_prompt(input)
    

class GPRFPromptTemplate(BasePromptTemplate):
    base_system_prompt = "Please rewrite the user's query based on several relevant passages (which may contain noise or errors). The rewritten query should preserve the original meaning while incorporating as much information as possible, so that search engines can more effectively retrieve relevant passages."
    base_user_prompt = "Relevant Passages:\n{documents}\nUser Query: {question}\n\nRewritten Query:"
    
    def __init__(self, config, enable_chat=True):
        super().__init__(config)
        self.enable_chat = enable_chat
        
        self.system_prompt = self.base_system_prompt
        self.user_prompt = self.base_user_prompt
        
    def get_string(self, question : str, passages : List[str], **params):
        input_params = {"question": question, "documents": self.format_reference(passages)}
        input_params.update(**params)
        
        system_prompt = self.system_prompt.format(**input_params)
        user_prompt = self.user_prompt.format(**input_params)

        if self.enable_chat:
            input = []
            if system_prompt != "":
                input.append({"role": "system", "content": system_prompt})
            if user_prompt != "":
                input.append({"role": "user", "content": user_prompt})
            
            input = self.tokenizer.apply_chat_template(input, tokenize=False, add_generation_prompt=True)
        else:
            input = "\n\n".join([prompt for prompt in [system_prompt, user_prompt] if prompt != ""])

        return self.truncate_prompt(input)
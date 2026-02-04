from __future__ import annotations
import math
import os
from typing import Dict, List
import numpy as np
from dotenv import load_dotenv

from langchain_community.llms import Ollama

from llm.prompt_generator import PromptGenerator
from llm.utils import get_program_str_from_llm_response_dsl, get_program_str_from_llm_response_python
from prog_policies.utils import get_env_name
from prog_policies.base import BaseDSL, dsl_nodes

# Load environment variables (kept in case you have other keys, but HF_TOKEN is no longer needed)
load_dotenv()

model_id = "qwen2.5:1.5b" 

class LLMProgramGenerator:
    def __init__(
        self,
        seed: int,
        task: str,
        dsl: BaseDSL,
        llm_program_num: int,
        temperature: float = 1.0,
        top_p: float = 0.95, # Ollama defaults slightly differently, adjusted to 0.95 usually
        action_shots: int = 0,
        perception_shots: int = 0,
        program_shots: int = 0,
    ) -> None:
        self.seed = seed
        self.task = task
        self.env_name = get_env_name(task)
        self.dsl = dsl
        self.ratio = 1.5
        self.llm_program_num = llm_program_num
        self.temperature = temperature
        self.top_p = top_p

        self.np_rng = np.random.RandomState(self.seed)

        # Revision parameters
        self.action_shots = action_shots
        self.perception_shots = perception_shots
        self.program_shots = program_shots
        self.prompt_generator = PromptGenerator(
            self.task,
            self.action_shots,
            self.perception_shots,
            self.program_shots,
        )
        
        self.model_name = model_id

        self.llm = Ollama(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            # We explicitly set stop tokens if needed, but usually not required for qwen
            # num_predict=1024  # Equivalent to max_new_tokens
        )

    def _call_llm(self, system_prompt: str, user_prompt: str, llm_program_num: int) -> list[str]:
        
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # Note: Ollama via LangChain processes these sequentially. 
        # If llm_program_num is 50, it will send 50 requests one by one.
        # On a local machine, this is usually safer than batching to avoid RAM crashes.
        
        print(f"Generating {llm_program_num} programs with {self.model_name}...")
        
        # We generate a list of prompts (identical) to simulate the 'n' generations
        generations = self.llm.generate([full_prompt] * llm_program_num).generations
        
        # Return a list of strings
        return [gen[0].text for gen in generations]

    # --------------------------------------------------------------------------
    # The rest of the methods below are unchanged logic-wise, 
    # just standard parsing of the results.
    # --------------------------------------------------------------------------

    def _get_program_list_from_llm_response_python_to_dsl(self, response) -> list[str]:
        program_str_list = []
        for x in response:
            tmp = []
            try:
                program_str = get_program_str_from_llm_response_python(x, env_name=self.env_name)
                tmp.append(program_str)
            except:
                pass
            
            try:
                program_str = get_program_str_from_llm_response_dsl(x, env_name=self.env_name)
                tmp.append(program_str)
            except:
                pass
            
            program_str_list.append(tmp)
        return program_str_list
    
    
    def _get_program_list_from_llm_response_python(self, response) -> list[str]:
        program_str_list = []
        for x in response:
            try:
                program_str = get_program_str_from_llm_response_python(x, env_name=self.env_name)
                program_str_list.append(program_str)
            except:
                pass
        return program_str_list
    
    def _get_program_list_from_llm_response_dsl(self, response) -> list[str]:
        program_str_list = []
        for x in response:
            try:
                program_str = get_program_str_from_llm_response_dsl(x, env_name=self.env_name)
                program_str_list.append(program_str)
            except:
                pass
        return program_str_list

    def get_program_list_python_to_dsl(self) -> tuple[list, dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            
            # Calculate how many to request
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_python_to_dsl()
            
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass # Consider printing e here for debugging
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
                    
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            
            # Convert back to strings for the log
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log
    
    # ... (Repeat the same minor logging updates for the other methods if desired, 
    # but the logic below works exactly as is because it calls _call_llm) ...

    def get_program_list_python(self) -> tuple[list, dict]:
        # Identical structure to original, just uses the new self.llm
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python()
            user_prompt = self.prompt_generator.get_user_prompt_python()
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python(llm_response)
            for program_str in program_str_list:
                try:
                    program = self.dsl.parse_str_to_node(program_str)
                    program_list.append(program)
                except:
                    pass
                
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log
    
    def get_program_list_dsl(self) -> tuple[list, dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_dsl()
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_dsl(llm_response)
            for program_str in program_str_list:
                try:
                    program = self.dsl.parse_str_to_node(program_str)
                    program_list.append(program)
                except:
                    pass
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]    
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log

    def get_program_list_revision_regeneration_with_reward(
        self,
        progs_rewards: list[dsl_nodes.Program, float]
    ):
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_regeneration_with_reward(progs_rewards, self.dsl)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log

    def get_program_list_revision_regeneration(
        self,
        previous_program_list: List[dsl_nodes.Program],
    ):
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_regeneration(previous_program_list, self.dsl)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log
    
    def get_program_list_revision_agent_execution_trace(
        self,
        reward: float, logs: list[dict[str, str]], average_reward: float,
    ) -> tuple[list[dsl_nodes.Program], dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_agent_execution_trace(reward, logs, average_reward)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )

        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log
    
    def get_program_list_revision_agent_program_execution_trace(
        self,
        reward: float, logs: list[dict[str, str]], average_reward: float,
    ) -> tuple[list[dsl_nodes.Program], dict]:
        program_list = []
        record_list = []
        attempts = 0
        program_num = self.llm_program_num
        while len(program_list) < program_num:
            attempts += 1
            seed = self.np_rng.randint(0, 2**32)
            llm_program_num = math.ceil((program_num - len(program_list)) * self.ratio)
            system_prompt = self.prompt_generator.get_system_prompt_python_to_dsl()
            user_prompt = self.prompt_generator.get_user_prompt_revision_agent_program_execution_trace(reward, logs, average_reward)
            llm_response = self._call_llm(system_prompt, user_prompt, llm_program_num)
            program_str_list = self._get_program_list_from_llm_response_python_to_dsl(llm_response)
            for candidates in program_str_list:
                tmp = []
                for candidate in candidates:
                    try:
                        program = self.dsl.parse_str_to_node(candidate)
                        tmp.append(program)
                    except:
                        pass
                if len(tmp) > 0:
                    program_list.append(self.np_rng.choice(tmp))
            
            available_program_num = len(program_list)
            print(f"Attempts: {attempts}, Programs Found: {available_program_num}")
            
            if len(program_list) > program_num:
                program_list = program_list[:program_num]
            program_str_list_log = [self.dsl.parse_node_to_str(program) for program in program_list]  
            
            record_list.append(
                {
                    "seed": seed,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "llm_response": llm_response,
                    "available_program_num": available_program_num,
                    "program_str_list": program_str_list_log,
                }
            )
        log = {"attemps": attempts, "record_list": record_list}
        return program_list, log
    
    
   

from retry import retry
from numpy import ndarray
from typing import List, Union
from string import ascii_uppercase
from abc import ABC, abstractmethod
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser #new
from pydantic import ValidationError, BaseModel #new

from utils.map_logger import MapLogger
from utils.panovis import PanoVisualizer
from utils.operation import image_to_base64, validate_choice_parsed
from utils.items import ViewPoint, ViewPointAttrToUpdate, StopReactNode, ChoiceReActNode, PanoItem

import json #new
import re #new
from langchain_core.outputs import Generation #new
from typing import Generic, List, Type, TypeVar #new


TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class PyOutputParser(PydanticOutputParser[TBaseModel]):
    def parse_result(self, result: List[Generation], *, partial: bool = False) -> TBaseModel:
        print("Parsing result...")

        # Step 1: 尝试直接解析，不调用 fix_json_format
        json_str = result[0].text  # Extract the text from the Generation result
        print("Original input before any fixing:", json_str)  # 打印原始输入
        
        try:
            print("Trying to directly parse the output...")
            parsed_object = self.pydantic_object.parse_obj(json.loads(json_str))
            print("????????????????????????????Successfully parsed JSON directly!")
            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!Direct parsing failed: {e}")
        
        # Step 2: 如果直接解析失败，调用 fix_json_format 修正格式
        try:
            print("Attempting to fix JSON format...")
            corrected_json = self._fix_json_format(json_str)
            parsed_object = self.pydantic_object.parse_obj(json.loads(corrected_json))
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAASuccessfully parsed JSON after fixing format!")
            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Parsing after format fixing failed: {e}")
            # Step 3: 如果修正后仍然失败，抛出错误
            raise ValidationError("BBBBBBBBBBBBBBBBBBBBBBBBBBBBFailed to parse JSON after attempting to fix formatting") from e

        
    def parse(self, text: str) -> TBaseModel:
        print("????????????????????????Parsing text...")
        return self.parse_result([Generation(text=text)], partial=False)
    
    def _fix_json_format(self, json_str: str) -> str:
        # Remove all Markdown-like code block enclosures 
        # Step 1: 去除三重反引号和可选的语言标记（如```json、```perl等）
        json_str = re.sub(r'```(?:\w+)?', '', json_str)
        json_str = json_str.strip()
        # Step 2: 尝试提取 JSON 对象，确保只保留从 `{` 开始到 `}` 结束的内容
        # Find the valid JSON structure
        json_start = json_str.find('{')
        json_end = json_str.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = json_str[json_start:json_end]
        # Step 3: 修正常见的 JSON 格式错误
        # 修正逗号错误，删除 `}` 和 `]` 前的多余逗号
        # Fix misplaced commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        # json_str = re.sub(r'("\s*:[^,}{\[\]]+)(\s*")', r'\1,\2', json_str)


        return json_str

# new
# import json
# import re
# from typing import List
# from pydantic import BaseModel, ValidationError
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.outputs import Generation

# class PyOutputParser(PydanticOutputParser):
#     def __init__(self, pydantic_object: BaseModel):
#         super().__init__(pydantic_object=pydantic_object)

#     def parse_result(self, result: List[Generation], *, partial: bool = False) -> BaseModel:
#         print("Parsing result...")

#         json_str = result[0].text  # Extract the text from the Generation result
#         print("Original input before any fixing:", json_str)

#         try:
#             print("Trying to directly parse the output...")
#             parsed_object = self.pydantic_object.parse_obj(json.loads(json_str))
#             print("Successfully parsed JSON directly!")
#             return parsed_object
#         except (ValidationError, json.JSONDecodeError) as e:
#             print(f"Direct parsing failed: {e}")

#         try:
#             print("Attempting to fix JSON format...")
#             corrected_json = self._fix_json_format(json_str)
#             parsed_object = self.pydantic_object.parse_obj(json.loads(corrected_json))
#             print("Successfully parsed JSON after fixing format!")
#             return parsed_object
#         except (ValidationError, json.JSONDecodeError) as e:
#             print(f"Parsing after format fixing failed: {e}")
#             raise ValidationError("Failed to parse JSON after attempting to fix formatting") from e

#     def parse(self, text: str) -> BaseModel:
#         print("Parsing text...")
#         return self.parse_result([Generation(text=text)], partial=False)

#     def _fix_json_format(self, json_str: str) -> str:
#         # Remove all Markdown-like code block enclosures 
#         json_str = re.sub(r'```(?:\w+)?', '', json_str).strip()
#         # Extract JSON object
#         json_start = json_str.find('{')
#         json_end = json_str.rfind('}') + 1
#         if json_start != -1 and json_end != -1:
#             json_str = json_str[json_start:json_end]
#         # Fix misplaced commas
#         json_str = re.sub(r',\s*}', '}', json_str)
#         json_str = re.sub(r',\s*\]', ']', json_str)
# #         return json_str    

class SpaceAgent(ABC):

    def __init__(self):
        self.agent: ChatOpenAI

    @property
    @abstractmethod
    def agent(self):
        pass

    @agent.setter
    @abstractmethod
    def agent(self, value):
        pass

    @abstractmethod
    def observe_and_think(self,
                          question: str,
                          viewpoint: ViewPoint,
                          backtracked: bool = False,
                          pred_action_on_start: int = 0
                          ) -> ViewPointAttrToUpdate:
        pass

    def _observe_pano(self,
                      image: Union[str, ndarray, property],
                      question: str,
                      backtracked: bool = False,
                      pred_action_on_start: int = 0
                      ) -> StopReactNode:
        image_dict = {
            "image_url": image_to_base64(image)
        }
        str_content = [
            {
                "type": "image_url", "image_url": {"url": "{image_url}"}
            }
        ]
        stop_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "#Instrunction:\nYou are a helpful robot to analyse images according peoples' question and help people "
                           "find the correct way to their destination, like find a bookstore and so on. Given the "
                           "question, you should point out that whether the scene contents could statisfy user's requirements"),
                ("user","#Output Format:\n{format_instructions}"), 
                ("user", """
                #Example:
                ##Input:
                    [
                        {{'type': 'text', 'text': 'I am hungry'}},
                        {{'type': 'image_url', 'image_url': '...'}}
                    ]
                ##Output:
                    {{
                        "observation": "There are some residential buildings, a bookstore, and a bus station in this image",
                        "thoughts": "I am hungry, so I should find a restaurant to have a meal. There is no restaurant here, so I should
                        keep going to find a restaurant."
                        "action": 0,
                    }}

                ##if you think user has already arrived at the destination, using 1 represents for stop, otherwise 0.
                ##Output:
                    {{
                        "thoughts": "I have already arrived at restaurant, where I can achieve my goal,
                         so I have to stop here. No movement needed anymore."
                        "observation": "There is a restaurant in this image"
                        "action": 1,
                    }}      

                #Now here it's your turn to help answer the question and output the result.
                ##Input:\n"""),
                ("user", "{query}"),
                ("user", "{backtrack_prompt}"),
                ("user", str_content)
            ]
        )
        # structured_llm = self.agent.with_structured_output(StopReactNode)
        chain = stop_prompt | structured_llm

        if backtracked:
            backtrack_prompt = f"Currently you just backtracked from image {ascii_uppercase[pred_action_on_start]}"
        else:
            backtrack_prompt = ""

        params = {
            "query": question,
            "backtrack_prompt": backtrack_prompt,
            "format_instructions":StopReactNode.model_json_schema()
        }
        params.update(image_dict)
        stop_react: StopReactNode = chain.invoke(params)

        return stop_react

    # @retry(exceptions=ValueError, tries=10)
    def _observe_perspective(self,
                             walkable_headings: List[float],
                             question: str,
                             ) -> ChoiceReActNode:
        params = {
            "fov": 120,
            "phi": 0,
            "height": 512,
            "width": 1024
        }

        images: List[PanoItem] = [
            PanoVisualizer.get_perspective(theta=heading, **params)
            for heading in walkable_headings
        ]

        str_content = []
        image_dict = {}

        for idx, image in enumerate(images):
            image_idx = f"{ascii_uppercase[idx]}"
            str_content.append({"type": "image_url", "image_url": {"url": f"{{{image_idx}}}"}})
            image_dict[image_idx] = image_to_base64(image.perspective)

        perspective_prompt = f"Here are {len(walkable_headings)} perspectives."
        MapLogger.info(perspective_prompt)

        choice_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "#Instrunction:\nYou are a helpful robot to analyse images according peoples' question and help people "
                           "find the correct way to their destination, like find a bookstore and so on. Given the "
                           "question, you should point out that which image is the most suitable as the answer using "
                           "its index like 0 with confidence score [0, 1]]."),
                ("user","#Output Format:\n{format_instructions}"),          
                ("user", """
                #Example:
                ##Input:
                    [
                        {{'type': 'text', 'text': 'I am hungry'}},
                        {{'type': 'image_url', 'image_url': '...'}},
                        {{'type': 'image_url', 'image_url': '...'}},
                    ]
                ##Output:
                    {{
                        "thoughts": "I am hungry, so I want to find the most potential road to go."
                        "observation": {{
                            "A": "there are a lot of cars.",
                            "B": "there are a lot of buildings."
                        }}
                        "action": 1,
                        "score": 0.78
                    }}

                # Now here it's your turn to help answer the question and output the index of the proper image.
                ## Input:\n"""),
                ("user", "Do attention that the action stands for the index of image, it starts from A, A for the first image, and B for the second image"),
                ("user", "{perspective_prompt}, your observations output should have the same num with the perspectives"),
                ("user", "{query}"),
                ("user", str_content),
                ("user", " ##Output:")
            ]
        )
        structured_llm = self.agent.with_structured_output(ChoiceReActNode)
        chain = choice_prompt | structured_llm

        params = {
            "query": question,
            "perspective_prompt": perspective_prompt,
            "format_instructions":ChoiceReActNode.model_json_schema()
        }
        params.update(image_dict)

        # choice_react: ChoiceReActNode = chain.invoke(params)
        max_try = 5
        for i in range(max_try):
            try:
                result = chain.invoke(params)
                print("####################","result invoke:",result)
                time.sleep(1)
                choice_react = ChoiceReActNode.model_validate(result)
                break
            except ValidationError  as e:
                MapLogger.error(f"invoke error:{result}")
        choice_react = validate_choice_parsed(choice_react, len(walkable_headings))

        return choice_react

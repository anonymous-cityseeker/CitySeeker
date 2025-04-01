from retry import retry
from numpy import ndarray
from typing import List, Union, Optional
from string import ascii_uppercase
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from utils.map_logger import MapLogger
from utils.panovis import PanoVisualizer
from utils.operation import image_to_base64, validate_choice_parsed, Compass
from utils.items import (
    ViewPoint, ViewPointAttrToUpdate,
    StopReactNode, ChoiceReActNode,
    PanoItem, ViewPointPosition, PanoParams
)



# 对了
import json
import re
from typing import List, Union, Optional
from pydantic import BaseModel, ValidationError
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.outputs import Generation
from langchain_core.exceptions import OutputParserException

class PyOutputParser(PydanticOutputParser):
    def __init__(self, pydantic_object: BaseModel):
        super().__init__(pydantic_object=pydantic_object)

    def parse_result(self, result: List[Generation], *, partial: bool = False) -> BaseModel:
        print("Parsing result...")

        json_str = result[0].text  # 从 Generation 结果中提取文本
        print("Original input before any fixing:", json_str)

        # 尝试直接解析
        try:
            print("Trying to directly parse the output...")
            parsed_object = self.pydantic_object.parse_obj(json.loads(json_str))
            print("Successfully parsed JSON directly!")
            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Direct parsing failed: {e}")

        # 尝试修复 JSON 格式
        try:
            print("Attempting to fix JSON format...")
            corrected_json = self._fix_json_format(json_str)
            print("Corrected JSON:", corrected_json)
            parsed_object = self.pydantic_object.parse_obj(json.loads(corrected_json))
            print("Successfully parsed JSON after fixing format!")
            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Parsing after format fixing failed: {e}")

        # 尝试将非 JSON 格式的键值对转换为 JSON
        try:
            print("Attempting to convert key-value pairs to JSON...")
            kv_json = self._convert_kv_to_json(json_str)
            print("Converted JSON:", kv_json)
            parsed_object = self.pydantic_object.parse_obj(json.loads(kv_json))
            print("Successfully parsed JSON after converting key-value pairs!")
            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Parsing after converting key-value pairs failed: {e}")
            # 抛出自定义的 OutputParserException
            raise OutputParserException("Failed to parse JSON after attempting to fix formatting and convert key-value pairs") from e

    def parse(self, text: str) -> BaseModel:
        print("Parsing text...")
        return self.parse_result([Generation(text=text)], partial=False)

    def _fix_json_format(self, json_str: str) -> str:
        # 移除所有 Markdown 代码块符号
        # json_str = re.sub(r'```(?:\w+)?', '', json_str).strip()
        json_str = re.sub(r'^```(?:\w+)?\n?', '', json_str).strip()
        # 移除结尾的Markdown代码块符号 new
        json_str = re.sub(r'\n?```$', '', json_str).strip()
        # 移除 "Output:" 行（如果存在）
        json_str = re.sub(r'^Output:\s*', '', json_str, flags=re.IGNORECASE)
        # 提取 JSON 对象
        json_start = json_str.find('{')
        json_end = json_str.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = json_str[json_start:json_end]         
        # 修正多余的逗号
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*\]', ']', json_str)
        return json_str

    
    # def _convert_kv_to_json(self, kv_str: str) -> str:
    #     """
    #     将类似于以下格式的键值对字符串转换为JSON对象：
    #     Observation: ...
    #     Thoughts: ...
    #     Action: ...
    #     Score: ...
    #     """
    #     kv_dict = {}
    #     # 使用正则表达式提取键值对

    #     pattern = re.compile(r'(\w+):\s*(.+)')
    #     # pattern = re.compile(r'(\w+):\s*(["\']?)(.*?)\2$')
    #     for line in kv_str.splitlines():
    #         match = pattern.match(line.strip())
    #         if match:
    #             key, value = match.groups()
    #             # 处理嵌套的JSON对象
    #             if value.startswith('{') and value.endswith('}'):
    #                 try:
    #                     value = json.loads(value.replace("'", '"'))
    #                 except json.JSONDecodeError:
    #                     pass  # 保持原始字符串
    #             else:
    #                 # 移除可能的结尾逗号
    #                 value = value.rstrip(',').strip()
    #                 # 去除可能的引号
    #                 value = value.strip('"').strip("'")
    #             kv_dict[key.lower()] = value
    #     # 转换为JSON字符串
    #     return json.dumps(kv_dict)

    # def _convert_kv_to_json(self, kv_str: str) -> str:
    #     """
    #     将类似于以下格式的键值对字符串转换为JSON对象：
    #     Observation: ...
    #     Thoughts: ...
    #     Action: ...
    #     Score: ...
    #     """
    #     kv_dict = {}
    #     # 使用正则表达式提取键值对
    #     pattern = re.compile(r'(\w+):\s*(["\']?)(.*?)\2$')
    #     for line in kv_str.splitlines():
    #         match = pattern.match(line.strip())
    #         if match:
    #             key, quote, value = match.groups()
    #             # 处理嵌套的JSON对象
    #             if value.startswith('{') and value.endswith('}'):
    #                 try:
    #                     value = json.loads(value.replace("'", '"'))
    #                 except json.JSONDecodeError:
    #                     print(f"Failed to parse nested JSON for key '{key}': {value}")
    #                     pass  # 保持原始字符串
    #             else:
    #                 # 移除可能的结尾逗号
    #                 value = value.rstrip(',').strip()
    #                 # 去除可能的引号
    #                 value = value.strip('"').strip("'")
    #             kv_dict[key.lower()] = value
    #     # 填充缺失的字段
    #     required_fields = ["thoughts", "observation", "action", "score"]
    #     for field in required_fields:
    #         if field not in kv_dict:
    #             if field == "score":
    #                 kv_dict[field] = 0.5
    #             elif field == "action":
    #                 kv_dict[field] = ascii_uppercase[0]  # 'A'等
    #             elif field == "observation":
    #                 kv_dict[field] = {ascii_uppercase[0]: "Default observation."}
    #             else:
    #                 kv_dict[field] = "Default thoughts."
    #     # 转换为JSON字符串
    #     return json.dumps(kv_dict)

    def _convert_kv_to_json(self, kv_str: str) -> str:
        """
        将类似于以下格式的键值对字符串转换为JSON对象：
        Observation: ...
        Thoughts: ...
        Action: ...
        Score: ...
        """
        kv_dict = {}
        # 使用正则表达式提取键值对
        pattern = re.compile(r'(\w+):\s*(["\']?)(.*?)\2$')
        for line in kv_str.splitlines():
            match = pattern.match(line.strip())
            if match:
                key, quote, value = match.groups()
                # 处理嵌套的JSON对象
                if value.startswith('{') and value.endswith('}'):
                    try:
                        value = json.loads(value.replace("'", '"'))
                    except json.JSONDecodeError:
                        print(f"Failed to parse nested JSON for key '{key}': {value}")
                        pass  # 保持原始字符串
                else:
                    # 移除可能的结尾逗号
                    value = value.rstrip(',').strip()
                    # 去除可能的引号
                    value = value.strip('"').strip("'")
                kv_dict[key.lower()] = value

        # 根据模型类型定义必需字段及其默认值
        required_fields = self._get_required_fields()
        for field in required_fields:
            if field not in kv_dict:
                if field == "score":
                    kv_dict[field] = 0.5
                elif field == "action":
                    kv_dict[field] = self._get_default_action()
                elif field == "observation":
                    kv_dict[field] = {ascii_uppercase[0]: "Default observation."}
                else:
                    kv_dict[field] = "Default thoughts."

        # 转换为JSON字符串
        return json.dumps(kv_dict)

    def _get_default_action(self):
        """
        根据当前的 pydantic_object 类型，返回 action 字段的默认值。
        """
        if issubclass(self.pydantic_object, StopReactNode):
            return 0  # 整数
        elif issubclass(self.pydantic_object, ChoiceReActNode):
            return ascii_uppercase[0]  # 字符串 'A'
        else:
            # 针对未知类型，抛出异常或返回一个安全的默认值
            raise OutputParserException(f"Unsupported pydantic_object type for action default: {type(self.pydantic_object)}")

    def _get_required_fields(self) -> List[str]:
        """
        根据当前的 pydantic_object 类型，返回必需字段列表。
        """
        if issubclass(self.pydantic_object, StopReactNode):
            return ["thoughts", "observation", "action"]  # 不需要 "score"
        elif issubclass(self.pydantic_object, ChoiceReActNode):
            return ["thoughts", "observation", "action", "score"]
        else:
            raise OutputParserException(f"Unsupported pydantic_object type for required fields: {type(self.pydantic_object)}")
# 以上    



class SpaceAgent(ABC):

    def __init__(self):
        self.agent: ChatOpenAI
        self._pano_params = PanoParams().to_dict(encode_json=True)  # noqa
        # self.stop_parser = PydanticOutputParser(pydantic_object=StopReactNode)
        # self.choice_parser = PydanticOutputParser(pydantic_object=ChoiceReActNode)
        self.stop_parser = PyOutputParser(pydantic_object=StopReactNode)
        self.choice_parser = PyOutputParser(pydantic_object=ChoiceReActNode)

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
                          pred_action_on_start: int = 0,
                          last_position: ViewPointPosition = None,
                          curr_position: ViewPointPosition = None,
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
        # stop_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "You are a helpful robot to analyse images according peoples' question and help people "
        #                    "find the correct way to their destination, like find a bookstore and so on. Given the "
        #                    "question, you should point out that whether the scene contents could statisfy user's requirements"),
        #         ("system", """
        #         For example.
        #         Input:
        #             [
        #                 {{'type': 'text', 'text': 'I am hungry'}},
        #                 {{'type': 'image_url', 'image_url': '...'}}
        #             ]
        #         Output:
        #             {{
        #                 "observation": "There are some residential buildings, a bookstore, and a bus station in this image",
        #                 "thoughts": "I am hungry, so I should find a restaurant to have a meal. There is no restaurant here, so I should
        #                 keep going to find a restaurant."
        #                 "action": 0,
        #             }}

        #         if you think user has already arrived at the destination, using 1 represents for stop, otherwise 0.
        #         For example.
        #         Output:
        #             {{
        #                 "thoughts": "I have already arrived at restaurant, where I can achieve my goal,
        #                  so I have to stop here. No movement needed anymore."
        #                 "observation": "There is a restaurant in this image"
        #                 "action": 1,
        #             }}      

        #         Now here it's your turn to help answer the question and output the result.
        #         Input:\n"""),
        #         ("system", "{query}"),
        #         ("system", "{backtrack_prompt}"),
        #         ("user", str_content)
        #     ]
        # )

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

        chain = stop_prompt | self.agent | self.stop_parser

        if backtracked:
            backtrack_prompt = f"Currently you just backtracked from image {ascii_uppercase[pred_action_on_start]}"
        else:
            backtrack_prompt = ""

        # params = {
        #     "query": question,
        #     "backtrack_prompt": backtrack_prompt
        # }


        params = {
            "query": question,
            "backtrack_prompt": backtrack_prompt,
            # "format_instructions":StopReactNode.model_json_schema()
            "format_instructions": self.stop_parser.get_format_instructions()
        }

        params.update(image_dict)

        # stop_react: StopReactNode = chain.invoke(params)

        # return stop_react
    
        try:
            stop_react: StopReactNode = chain.invoke(params)
            return stop_react
        except (OutputParserException, ValidationError, Exception) as e:
            # 处理解析异常，例如记录日志或返回默认值
            MapLogger.info(f"Parsing failed: {e}")
            # 选择一个默认动作，例如选择第一个选项，置信度较低
            default_stop = StopReactNode(
                thoughts="Unable to parse response, choosing default action.",
                observation={ascii_uppercase[0]: "Default observation."},
                action=0,  # 赋值为整数
            )
            return default_stop    

    @retry(exceptions=ValueError, tries=5)
    def _observe_perspective(self,
                             walkable_headings: List[float],
                             question: str,
                             last_position: Optional[ViewPointPosition] = None,
                             curr_position: ViewPointPosition = None,
                             ) -> ChoiceReActNode:
        forward_azimuth = 0
        if last_position:
            forward_azimuth = Compass.get_step_forward_azimuth(last_position, curr_position)

        images: List[PanoItem] = [
            PanoVisualizer.get_perspective(theta=heading, **self._pano_params)
            for heading in walkable_headings
        ]

        str_content = []
        image_dict = {}
        direction_prompt = {}

        for idx, image in enumerate(images):
            image_idx = f"{ascii_uppercase[idx]}"
            str_content.append({"type": "image_url", "image_url": {"url": f"{{{image_idx}}}"}})
            if last_position:
                direction_prompt[f"{image_idx}"] = f"This perspective is on your {Compass.get_relative_direction(forward_azimuth, walkable_headings[idx]).value}"
            image_dict[image_idx] = image_to_base64(image.perspective)

        perspective_prompt = f"Here are {len(walkable_headings)} perspectives."
        MapLogger.info(perspective_prompt)

        # choice_prompt = ChatPromptTemplate.from_messages(
        #     [
        #         ("system", "You are a helpful robot to analyse images according peoples' question and help people "
        #                    "find the correct way to their destination, like find a bookstore and so on. Given the "
        #                    "question, you should point out that which image is the most suitable as the answer using "
        #                    "its index like 0 with confidence score [0, 1]]."),
        #         ("system", """
        #         For example.
        #         Input:
        #             [
        #                 {{'type': 'text', 'text': 'I am hungry'}},
        #                 {{'type': 'image_url', 'image_url': '...'}},
        #                 {{'type': 'image_url', 'image_url': '...'}},
        #             ]
        #         Output:
        #             {{
        #                 "thoughts": "I am hungry, so I want to find the most potential road to go."
        #                 "observation": {{
        #                     "A": "there are a lot of cars.",
        #                     "B": "there are a lot of buildings."
        #                 }}
        #                 "action": 'A',
        #                 "score": 0.78
        #             }}

        #         Now here it's your turn to help answer the question and output the index of the proper image.
        #         Input:\n"""),
        #         ("system", "Do attention that the action stands for the index of image, it starts from A, A for the first image, and B for the second image"),
        #         ("system", "{perspective_prompt}, your observations output should have the same num with the perspectives"),
        #         ("system", "{direction_prompt}"),
        #         ("system", "give more opportunities to the way on the FRONT, LEFT, RIGHT side of you, try less or even do not step BACK"),
        #         ("system", "{query}"),
        #         ("user", str_content)
        #     ]
        # )


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
                        "action": "A",
                        "score": 0.78
                    }}

                # Now here it's your turn to help answer the question and output the index of the proper image.
                ## Input:\n"""),
                ("user", "Do attention that the action stands for the index of image, it starts from A, A for the first image, and B for the second image"),
                ("user", "{perspective_prompt}, your observations output should have the same num with the perspectives"),
                ("user", "{direction_prompt}"),
                ("user", "give more opportunities to the way on the FRONT, LEFT, RIGHT side of you, try less or even do not step BACK"),
                ("user", "{query}"),
                ("user", str_content),
                ("user", " ##Output:")
            ]
        )

        chain = choice_prompt | self.agent | self.choice_parser

        # params = {
        #     "query": question,
        #     "perspective_prompt": perspective_prompt,
        #     "direction_prompt": direction_prompt
        # }

        params = {
            "query": question,
            "perspective_prompt": perspective_prompt,
            "direction_prompt": direction_prompt,
            # "format_instructions":ChoiceReActNode.model_json_schema()
            "format_instructions": self.choice_parser.get_format_instructions()
        }

        params.update(image_dict)

        # choice_react: ChoiceReActNode = chain.invoke(params)
        # choice_react = validate_choice_parsed(choice_react, len(walkable_headings))

        # return choice_react
        
        try:
            choice_react: ChoiceReActNode = chain.invoke(params)
            choice_react = validate_choice_parsed(choice_react, len(walkable_headings))
            return choice_react
        # except OutputParserException as e:
        except (OutputParserException, ValidationError, Exception) as e:
            # 处理解析异常，例如记录日志或返回默认值
            MapLogger.info(f"Parsing failed: {e}")
            # 选择一个默认动作，例如选择第一个选项，置信度较低
            default_choice = ChoiceReActNode(
                thoughts="Unable to parse response, choosing default action.",
                observation={ascii_uppercase[0]: "Default observation."},
                action=ascii_uppercase[0],
                score=0.5
            )
            return default_choice

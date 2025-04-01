import json
import re

from typing import Optional, Type
from pydantic import BaseModel, ValidationError
from langchain_core.outputs import Generation
from langchain_core.utils.pydantic import TBaseModel
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser

from utils.items import ReActNode
from utils.map_logger import logger


class CityWalkerParser(PydanticOutputParser):

    def __init__(self, pydantic_object):
        super().__init__(pydantic_object=pydantic_object)

    def parse_result(
            self, result: list[Generation], *, partial: bool = False
    ) -> Optional[TBaseModel]:
        logger.debug(f"Parsing result...")

        json_str = result[0].text
        print("Original input before any fixing:", json_str)
        logger.debug(f"Original input before any fixing: {json_str}")

        try:
            logger.debug("Trying to directly parse the output...")
            parsed_object = self.pydantic_object.parse_obj(json.loads(json_str))
            logger.debug("Successfully parsed JSON directly!")

            return parsed_object
        except (ValidationError, json.JSONDecodeError) as e:
            logger.debug(f"Direct parsing failed: {e}")

        try:
            logger.debug("Attempting to fix JSON format...")
            corrected_json = self._fix_json_format(json_str)
            print("Corrected JSON:", corrected_json)
            logger.debug(f"Corrected JSON: {corrected_json}")

            parsed_object = self.pydantic_object.parse_obj(json.loads(corrected_json))

            logger.debug("Successfully parsed JSON after fixing format!")
            return parsed_object

        except (ValidationError, json.JSONDecodeError) as e:
            logger.debug(f"Parsing after format fixing failed: {str(e)}")

        # 尝试将非 JSON 格式的键值对转换为 JSON
        try:
            logger.debug("Attempting to convert key-value pairs to JSON...")
            kv_json = self._convert_kv_to_json(json_str)
            print("Converted JSON:", kv_json)
            logger.debug(f"Converted JSON: {kv_json}")

            parsed_object = self.pydantic_object.parse_obj(json.loads(kv_json))
            logger.debug("Successfully parsed JSON after converting key-value pairs!")
            return parsed_object

        except (ValidationError, json.JSONDecodeError) as e:
            logger.debug(f"Parsing after converting key-value pairs failed: {e}")
            # 抛出自定义的 OutputParserException
            raise OutputParserException(
                "Failed to parse JSON after attempting to fix formatting and convert key-value pairs") from e

    def parse(self, text: str) -> BaseModel:
        logger.debug("Parsing text...")
        return self.parse_result([Generation(text=text)], partial=False)

    @classmethod
    def _fix_json_format(cls, json_str: str) -> str:
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

    @classmethod
    def _convert_kv_to_json(cls, kv_str: str) -> str:
        """
        将类似于以下格式的键值对字符串转换为JSON对象：
        Observation: ...
        Thoughts: ...
        Action: ...
        Score: ...
        """
        kv_dict = {}
        # 使用正则表达式提取键值对

        pattern = re.compile(r'(\w+):\s*(.+)')
        # pattern = re.compile(r'(\w+):\s*(["\']?)(.*?)\2$')
        for line in kv_str.splitlines():
            match = pattern.match(line.strip())
            if match:
                key, value = match.groups()
                # 处理嵌套的JSON对象
                if value.startswith('{') and value.endswith('}'):
                    try:
                        value = json.loads(value.replace("'", '"'))
                    except json.JSONDecodeError:
                        pass  # 保持原始字符串
                else:
                    # 移除可能的结尾逗号
                    value = value.rstrip(',').strip()
                    # 去除可能的引号
                    value = value.strip('"').strip("'")
                kv_dict[key.lower()] = value
        # 转换为JSON字符串
        return json.dumps(kv_dict)







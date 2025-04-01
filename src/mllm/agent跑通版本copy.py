from retry import retry
from numpy import ndarray
from typing import List, Union
from string import ascii_uppercase
from abc import ABC, abstractmethod
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from utils.map_logger import MapLogger
from utils.panovis import PanoVisualizer
from utils.operation import image_to_base64, validate_choice_parsed
from utils.items import ViewPoint, ViewPointAttrToUpdate, StopReactNode, ChoiceReActNode, PanoItem


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
        structured_llm = self.agent.with_structured_output(StopReactNode)
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

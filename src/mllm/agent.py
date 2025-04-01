from retry import retry
from numpy import ndarray
from typing import List, Union, Optional
from string import ascii_uppercase
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from utils.map_logger import logger
from utils.panovis import PanoVisualizer
from utils.parser import CityWalkerParser
from utils.operation import image_to_base64, validate_choice_parsed, Compass
from utils.items import (
    ViewPoint, ViewPointAttrToUpdate,
    StopReactNode, ChoiceReActNode,
    PanoItem, ViewPointPosition, PanoParams
)


class SpaceAgent(ABC):

    def __init__(self):
        self.agent: ChatOpenAI
        self._pano_params = PanoParams().to_dict(encode_json=True)  # noqa
        self.stop_parser = CityWalkerParser(pydantic_object=StopReactNode)
        self.choice_parser = CityWalkerParser(pydantic_object=ChoiceReActNode)

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
                          last_position: ViewPointPosition = None,
                          curr_position: ViewPointPosition = None,
                          last_forward_azimuth: float = None,
                          **kwargs
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

        chain = stop_prompt | self.agent | self.stop_parser

        if backtracked:
            backtrack_prompt = f"Currently you just backtracked from image {ascii_uppercase[pred_action_on_start]}"
        else:
            backtrack_prompt = ""

        params = {
            "query": question,
            "backtrack_prompt": backtrack_prompt,
            "format_instructions": self.stop_parser.get_format_instructions()
        }
        params.update(image_dict)

        try:
            stop_react: StopReactNode = chain.invoke(params)
            return stop_react
        except Exception as e:

            logger.error(f"Parsing failed: {str(e)}, set default value.")

            default_stop = StopReactNode(
                thoughts="Unable to parse response, choosing default action.",
                observation="Default observation.",
                action=0,
            )
            return default_stop

    @retry(exceptions=ValueError, tries=5, logger=logger)
    def _observe_perspective(self,
                             walkable_headings: List[float],
                             question: str,
                             last_position: Optional[ViewPointPosition] = None,
                             curr_position: ViewPointPosition = None,
                             **kwargs
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
                direction_prompt[
                    f"{image_idx}"] = f"This perspective is on your {Compass.get_relative_direction(forward_azimuth, walkable_headings[idx]).value}"
            image_dict[image_idx] = image_to_base64(image.perspective)

        perspective_prompt = f"Here are {len(walkable_headings)} perspectives."
        logger.info(perspective_prompt)

        choice_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", "#Instrunction:\nYou are a helpful robot to analyse images according peoples' question and help people "
                         "find the correct way to their destination, like find a bookstore and so on."
                         "Given the question, you should point out that which image is the most suitable as the answer using "
                         "its index like 0 with confidence score [0, 1]]."),
                ("user", "#Output Format:\n{format_instructions}"),
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
                ("user", "you just backtracked, and you should select the correct image index {index}") if kwargs["backtracked"] == True else '',  # TODO optimize prompt
                ("user", "For the same question, I have previously visited this location multiple rounds, gathered surrounding information, recorded my choices, and noted whether I ultimately reached the destination: {surrounding_prompt}" if "retrieved_information" in kwargs.keys() else ''),  # TODO optimize prompt
                ("user", "The historical trajectory of my visits includes: {history_nodes_prompt}" if "history_nodes" in kwargs.keys() else ''),  # TODO


                ("user", "{query}"),
                ("user", str_content),
                ("user", " ##Output:")
            ]
        )
        chain = choice_prompt | self.agent | self.choice_parser

        params = {
            "query": question,
            "perspective_prompt": perspective_prompt,
            "direction_prompt": direction_prompt,
            "format_instructions": self.choice_parser.get_format_instructions(),
        }
        if kwargs["backtracked"]:
            params.update({
                "index": kwargs["prompt_perspective_idx"]
            })
        if "retrieved_information" in kwargs.keys():
            params.update({
                "surrounding_prompt": kwargs['retrieved_information'] # TODO
            })
        if "history_nodes_prompt" in kwargs.keys():
            params.update({
                "history_nodes_prompt": kwargs['history_nodes_prompt'] # TODO
            })



        params.update(image_dict)

        try:
            choice_react: ChoiceReActNode = chain.invoke(params)
            choice_react = validate_choice_parsed(choice_react, len(walkable_headings))

            return choice_react

        except Exception as e:

            logger.error(f"Parsing failed: {str(e)}, set default value.")

            default_choice = ChoiceReActNode(
                thoughts="Unable to parse result from response, choosing default action.",
                observation={ascii_uppercase[0]: "Default observation."},
                action=ascii_uppercase[0],
                score=0.5
            )
            return default_choice

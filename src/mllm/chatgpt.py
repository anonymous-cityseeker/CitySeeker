import os
import json

from string import ascii_uppercase

from langchain_openai import ChatOpenAI

from src.mllm.agent import SpaceAgent
from utils.operation import Compass
from utils.panovis import PanoVisualizer
from utils.items import ViewPoint, ViewPointAttrToUpdate, VisitStatus, StopStatus, ViewPointPosition


# class ChatGPT4o(SpaceAgent):

#     OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#     def __init__(self):
#         super().__init__()

#         self._agent = ChatOpenAI(model_name="chatgpt-4o",
#                                  openai_api_key=self.OPENAI_API_KEY,
#                                  openai_api_base=self.OPENAI_API_BASE)

class ChatGPT4o(SpaceAgent):

    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME")
   
    def __init__(self):
        super().__init__()
        print(self.OPENAI_API_BASE)
        print(self.MODEL_NAME)
        # import pdb
        # pdb.set_trace()
        self._agent = ChatOpenAI(model_name=self.MODEL_NAME,
                                 api_key=self.OPENAI_API_KEY,
                                 base_url=self.OPENAI_API_BASE)
    @property
    def agent(self):
        return self._agent

    def observe_and_think(self,
                          question: str,
                          viewpoint: ViewPoint,
                          last_position: ViewPointPosition = None,
                          curr_position: ViewPointPosition = None,
                          last_forward_azimuth: float = None,
                          **kwargs
                          ) -> ViewPointAttrToUpdate:
        PanoVisualizer.set_pano(viewpoint.filename)
        PanoVisualizer.set_heading(viewpoint.heading)

        pano_react = self._observe_pano(PanoVisualizer.PANO, question)
        choice_react = self._observe_perspective(
            viewpoint.walkable_headings, question,
            last_position, curr_position, **kwargs
        )

        # neo4j can not store map values.
        if isinstance(choice_react.observation, dict):
            choice_react.observation = json.dumps(choice_react.observation)

        if isinstance(pano_react.observation, dict):
            pano_react.observation = json.dumps(pano_react.observation)

        if pano_react.action == StopStatus.STOP.value:  # if stopped
            action = StopStatus.STOP.REACHED.value
        else:
            action = ascii_uppercase.index(choice_react.action.upper())

        if last_position:
            direction = Compass.get_relative_direction(last_forward_azimuth,
                                                       viewpoint.walkable_headings[pano_react.action]).value
        else:
            direction = "<START>"

        return ViewPointAttrToUpdate(
            filename=viewpoint.filename,
            observations=pano_react.observation,
            perspective_observation=choice_react.observation,
            thought=pano_react.thoughts,
            score=choice_react.score,
            pred_action=action,
            visited=VisitStatus.CURRENT_VISITED,
            action_direction=direction
        )

import os
import json

from string import ascii_uppercase
from langchain_openai import ChatOpenAI

from src.mllm.agent import SpaceAgent
from utils.panovis import PanoVisualizer
from utils.items import ViewPoint, ViewPointAttrToUpdate, VisitStatus, StopStatus, ViewPointPosition


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
    
    def _format_observation(self, observation):
        # 统一将字典类型的 observation 转换为 JSON 字符串
        if isinstance(observation, dict):
            return json.dumps(observation)
        return observation
        

    def observe_and_think(self,
                          question: str,
                          viewpoint: ViewPoint,
                          backtracked: bool = False,
                          pred_action_on_start: int = 0,
                          last_position: ViewPointPosition = None,
                          curr_position: ViewPointPosition = None,
                          ) -> ViewPointAttrToUpdate:
        PanoVisualizer.set_pano(viewpoint.filename)
        PanoVisualizer.set_heading(viewpoint.heading)

        pano_react = self._observe_pano(PanoVisualizer.PANO, question)
        choice_react = self._observe_perspective(
            viewpoint.walkable_headings, question,
            last_position, curr_position
        )

        # neo4j can not store map values.
        # if isinstance(choice_react.observation, dict):
        #     choice_react.observation = json.dumps(choice_react.observation)
        
        # 使用统一的转换方法来格式化 observations
        pano_react.observation = self._format_observation(pano_react.observation)
        choice_react.observation = self._format_observation(choice_react.observation)


        if pano_react.action == StopStatus.STOP.value:  # if stopped
            action = StopStatus.STOP.REACHED.value
        else:
            action = ascii_uppercase.index(choice_react.action.upper())

        return ViewPointAttrToUpdate(
            filename=viewpoint.filename,
            observations=pano_react.observation,
            perspective_observation=choice_react.observation,
            thought=pano_react.thoughts,
            score=choice_react.score,
            pred_action=action,
            visited=VisitStatus.CURRENT_VISITED
        )

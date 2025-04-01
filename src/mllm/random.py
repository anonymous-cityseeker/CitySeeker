import numpy as np
from src.mllm.agent import SpaceAgent
from utils.items import ViewPoint, ViewPointPosition, ViewPointAttrToUpdate, ChoiceReActNode, VisitStatus
from utils.operation import Compass


class RandomBaselineModel(SpaceAgent):

    def __init__(self):
        super().__init__()
        self._agent = None

    @property
    def agent(self):
        return self._agent

    def observe_and_think(self,
                          question: str,
                          viewpoint: ViewPoint,
                          last_position: ViewPointPosition = None,
                          curr_position: ViewPointPosition = None,
                          last_forward_azimuth: float = None, **kwargs) -> ViewPointAttrToUpdate:

        pred_action = np.random.choice(range(len(viewpoint.walkable_headings)))

        if last_position:
            direction = Compass.get_relative_direction(last_forward_azimuth,
                                                       viewpoint.walkable_headings[pred_action]).value
        else:
            direction = "<START>"

        return ViewPointAttrToUpdate(
            filename=viewpoint.filename,
            observations="",
            perspective_observation="",
            thought="",
            score=0.5,
            pred_action=pred_action,
            visited=VisitStatus.CURRENT_VISITED,
            action_direction=direction
        )

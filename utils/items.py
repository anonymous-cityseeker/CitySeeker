from enum import Enum
from numpy import ndarray
from typing import List, Union, Dict
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config


class VisitStatus(Enum):
    UNVISITED = "UNVISITED"
    CURRENT_VISITED = "CURRENT_VISITED"
    HISTORY_VISITED = "HISTORY_VISITED"


class StopStatus(Enum):
    STOP = 1
    CONTINUE = 0
    REACHED = -1


class Direction(Enum):
    FRONT = "FRONT"
    FRONT_LEFT = "FRONT_LEFT"
    LEFT = "LEFT"
    BACK_LEFT = "BACK_LEFT"
    BACK = "BACK"
    BACK_RIGHT = "BACK_RIGHT"
    RIGHT = "RIGHT"
    FRONT_RIGHT = "FRONT_RIGHT"

    @classmethod
    def get_direction_by_index(cls, idx: int):
        return list(cls)[idx]

class MultiModels(Enum):
    """
    MultiModels
    """
    # S Models (7-16B)
    InternVL2_5_8B = "InternVL2_5_8B"
    llama3_llava_next_8b_hf = "llama3_llava_next_8b_hf"
    llava_onevision_qwen2_7b_si_hf = "llava_onevision_qwen2_7b_si_hf"
    MiniCPM_V_2_6 = "MiniCPM_V_2_6"
    Phi_3_5_vision_instruct = "Phi_3_5_vision_instruct"
    gpt_4o_mini = "gpt_4o_mini"
    Qwen2_VL_7B_Instruct = "Qwen2_VL_7B_Instruct"
    Llama_3_2_11B_Vision = "Llama_3_2_11B_Vision"

    # M Models (26-38B)
    InternVL2_5_26B = "InternVL2_5_26B"
    InternVL2_5_38B = "InternVL2_5_38B"
    deepseek_vl2 = "deepseek_vl2"

    # L Models (70-90B)
    ChatGPT4o = "ChatGPT4o"
    Qwen2_VL_72B_Instruct = "Qwen2_VL_72B_Instruct"
    InternVL2_5_78B = "InternVL2_5_78B"
    llama3_2_90b = "llama3_2_90b"
    gemini_1_5_pro = "gemini_1_5_pro"
    claude_3_5_sonnet = "claude_3_5_sonnet"
    MiniMax_01 = "MiniMax_01"

    # Uncertain Models
    gemini_2_0_flash_exp = "gemini_2_0_flash_exp"

    StraightBaselineModel = "StraightBaselineModel"
    RandomBaselineModel = "RandomBaselineModel"



class ReActNode(BaseModel):
    thoughts: str = Field(description="thoughts on every street view image")


class ChoiceReActNode(ReActNode):
    """
    used to select which image using A, B, C to stands for
    """
    action: str = Field(description="index of image selected. It starts from A, A stands for the first image",
                        examples=["A", "B", "C", "D"])
    score: float = Field(description="confidence score of image choice", examples=[0.78])
    observation: Union[str, dict] = Field(
        description="observation on every perspect of street view image",
        examples=[{"A": "there are a lot of cars", "B": "there are a lot of buildings"}]
    )  # problem happens when dict, using str first, then json.loads


class StopReactNode(ReActNode):
    """
    think whether to continue or stop
    """
    action: int = Field(
        description="0 for continue, 1 for stop",
        examples=[
            StopStatus.STOP.value,
            StopStatus.CONTINUE.value
        ]
    )
    observation: Union[dict, str] = Field(description="observation on every street view image")


@dataclass
class ViewPoint:
    filename: str = None
    heading: float = None
    walkable_headings: List[float] = None


@dataclass_json
@dataclass
class ViewPointPosition:
    filename: str = None
    longitude: float = None
    latitude: float = None



@dataclass
class PanoItem:
    perspective: ndarray
    mask: ndarray


@dataclass
class NodeChoice:
    react: Union[ChoiceReActNode, StopReactNode]
    image: PanoItem


@dataclass_json
@dataclass
class ViewPointAttrToUpdate:
    # index
    filename: str = None

    # observations
    observations: str = None
    perspective_observation: str = None

    # thoughts
    thought: str = None

    # action
    score: float = None
    visited: VisitStatus = None
    pred_action: int = None
    action_direction: str = None

@dataclass_json
@dataclass
class Trajectory:
    visited_nodes: List[Dict[str, ViewPointAttrToUpdate]] = None
    flag: bool = False


@dataclass_json
@dataclass
class GroundTruthTrajectories:
    @dataclass_json
    @dataclass
    class SingleTrajectory:
        question: str = None
        question_idx: int = None
        idx: int = None
        _from: str = field(default=None, metadata=config(field_name="from"))
        to: str = None
        service: str = None
        total_weight: float = None
        total_steps: int = None
        complete_route: List[ViewPointPosition] = field(default_factory=lambda: [ViewPointPosition()])

    data: List[SingleTrajectory] = field(default_factory=lambda: [GroundTruthTrajectories.SingleTrajectory()])


@dataclass_json
@dataclass
class SimulationTrajectories:
    @dataclass_json
    @dataclass
    class Trajectory(GroundTruthTrajectories.SingleTrajectory):
        flag: bool = False
        cost: float = 0.0
        round: int = 0
        round_success: bool = False
    data: List[Trajectory] = field(default_factory=lambda: [SimulationTrajectories.Trajectory()])


@dataclass_json
@dataclass
class PanoParams:
    fov: int = 120
    phi: float = 0.0
    height: int = 512
    width: int = 1024


@dataclass_json
@dataclass
class ViewPointPositionWithObservation(ViewPointPosition):
    overall_observation: str = None
    perspective_observation: Union[dict, str] = None
    thoughts: str = None
    action: int = None
    action_direction: str = None
    score: float = None

    @classmethod
    def from_viewpoint_position(cls,
                                viewpoint_position: ViewPointPosition,
                                update_viewpoint: ViewPointAttrToUpdate
                        ):
        return cls(
            filename=viewpoint_position.filename,
            longitude=viewpoint_position.longitude,
            latitude=viewpoint_position.latitude,
            overall_observation=update_viewpoint.observations,
            perspective_observation=update_viewpoint.perspective_observation,
            thoughts=update_viewpoint.thought,
            action=update_viewpoint.pred_action,
            action_direction=update_viewpoint.action_direction,
            score=update_viewpoint.score
        )

@dataclass_json
@dataclass
class LastStepMemory:

    last_step_filename: str = None,
    last_action: int = None
    last_action_direction: str = None
    last_score: float = None
    last_forward_azimuth: float = None

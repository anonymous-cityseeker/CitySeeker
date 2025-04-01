from __future__ import annotations

import json
import os

from tqdm import tqdm
from queue import Queue
from pathlib import Path
from datetime import datetime
from typing import Union, Literal
from loguru._logger import Logger, Core
from utils.items import ViewPointPosition, SimulationTrajectories, ViewPointPositionWithObservation


class MapLogger(Logger):

    def __init__(self,
                 json_file: Union[str, Path],
                 city_name: str,
                 section: str,
                 agent: str,
                 log_level: Literal["INFO", "DEBUG"] = "INFO"
                 ):
        super().__init__(
            core=Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patchers=[],
            extra={},
        )
        _LOG_DIR = Path(__file__).parent.parent / "output_dir" / city_name / section / "logs"
        _LOG_DIR.mkdir(exist_ok=True, parents=True)
        _LOG_FILE = _LOG_DIR / f"log_{agent}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        self.remove()
        self.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level=log_level)
        self.add(_LOG_FILE.as_posix(), encoding="utf-8", level=log_level)
        self.info(f"Current simulation will be logged to file {_LOG_FILE.as_posix()}")

        self.trajectory: Queue[ViewPointPositionWithObservation] = Queue()
        self.json_file = json_file if isinstance(json_file, str) else json_file.as_posix()
        self.distance_container = []
        with open(self.json_file, mode="w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls,
                  json_file: Union[str, Path],
                  city_name: str,
                  section: str,
                  agent: str,
                  log_level: Literal["INFO", "DEBUG"] = "INFO"
                  ) -> MapLogger:
        return cls(
            json_file=json_file,
            city_name=city_name,
            section=section,
            agent=agent,
            log_level=log_level
        )

    def insert_step(self,
                    viewpoint: ViewPointPositionWithObservation,
                    distance: float) -> None:
        self.trajectory.put(viewpoint)
        self.distance_container.append(distance)

        self.info(f"""
        
                    Current Position: {viewpoint}
                    Step Distance: {distance}m
                    """)

    def make_single_trajectory(self,
                               question: str,
                               question_idx: int,
                               idx: int,
                               _from: str,
                               to: str,
                               service: str,
                               total_weight: float,
                               total_steps: int,
                               flag: bool = False,
                               cost: float = 0.0,
                               round_num: int = 0,
                               round_success: bool = False) -> None:
        trajectory = SimulationTrajectories.Trajectory(
            question, question_idx, idx, _from, to, service, total_weight, total_steps,
            list(self.trajectory.queue), flag, cost, round_num, round_success
        )

        # clear all positions.
        with self.trajectory.mutex:
            self.trajectory.queue.clear()
            self.distance_container.clear()

        with open(self.json_file, mode="r+", encoding="utf-8") as f:
            data: list = json.load(f)
            f.seek(0)
            f.truncate(0)
            data.append(trajectory.to_dict(encode_json=True))
            json.dump(data, f, indent=4, ensure_ascii=False)

        self.success(f"Trajectory {idx} saved.")


logger = MapLogger.from_json(
    json_file=os.environ["STORE_JSON"],
    city_name=os.environ["CITY_NAME"],
    section=os.environ["SECTION"],
    agent=os.environ["AGENT"],
    log_level=os.environ["LOG_LEVEL"]
)

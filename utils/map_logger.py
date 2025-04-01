from __future__ import annotations

import json
from typing import Union, Literal
from pathlib import Path

from tqdm import tqdm
from queue import Queue
from datetime import datetime
from loguru import logger as loguru_logger

from utils.items import ViewPointPosition, SimulationTrajectories


class MapLogger:

    _logger = None

    def __init__(self,
                 json_file: Union[str, Path]):
        self.trajectory: Queue[ViewPointPosition] = Queue()
        self.json_file = json_file if isinstance(json_file, str) else json_file.as_posix()
        self.distance_container = []
        with open(self.json_file, mode="w", encoding="utf-8") as f:
            json.dump([], f, indent=4, ensure_ascii=False)

    @classmethod
    def from_json(cls,
                  json_file: Union[str, Path]) -> MapLogger:
        return cls(
            json_file=json_file
        )

    def insert_step(self,
                    viewpoint: ViewPointPosition,
                    distance: float) -> None:
        self.trajectory.put(viewpoint)
        self.distance_container.append(distance)

        MapLogger.info(f"""
        
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
                               cost: float = 0.0) -> None:
        trajectory = SimulationTrajectories.Trajectory(
            question, question_idx, idx, _from, to, service, total_weight, total_steps,
            list(self.trajectory.queue), flag, cost
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

        MapLogger.success(f"Trajectory {idx} saved.")

    @classmethod
    def set_log_file(cls,
                     city_name: str,
                     section: str,
                     agent: str,
                     log_level: Literal["INFO", "DEBUG"] = "INFO"):
        _LOG_DIR = Path(__file__).parent.parent / "output_dir" / city_name / section / "logs"
        _LOG_DIR.mkdir(exist_ok=True, parents=True)

        _LOG_FILE = _LOG_DIR / f"log_{agent}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

        cls._logger = loguru_logger

        cls._logger.remove()
        cls._logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
        cls._logger.add(_LOG_FILE.as_posix(), encoding="utf-8", level=log_level)
        cls._logger.info(f"Current simulation will be logged to file {_LOG_FILE.as_posix()}")

    @classmethod
    def success(cls, text: str):
        cls._logger.success(text)

    @classmethod
    def info(cls, text: str):
        cls._logger.info(text)

    @classmethod
    def warning(cls, text: str):
        cls._logger.warning(text)

    @classmethod
    def error(cls, text: str):
        cls._logger.error(text)

    @classmethod
    def opt(cls, colors=True):
        return cls._logger.opt(colors=colors)

    @classmethod
    def debug(cls, text: str):
        return cls._logger.debug(text)

    @classmethod
    @property
    def logger(cls):
        return cls._logger
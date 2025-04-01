from __future__ import annotations

import importlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import List, Literal

from utils.client import Neo4jClient
from utils.map_logger import MapLogger
from utils.operation import read_json, check_env_variables
from utils.items import GroundTruthTrajectories, ViewPointPosition, MultiModels, StopStatus


class Map:

    @check_env_variables()
    def __init__(self,
                 db_name: str,
                 gt_json: str,
                 agent: MultiModels,
                 backtrack: bool = False,
                 backtrack_steps: int = 0,
                 backtrack_threshold: float = 0.75,
                 log_level: Literal["INFO", "DEBUG"] = "INFO"
                 ):
        section = Path(gt_json).parent.name
        MapLogger.set_log_file(city_name=db_name, section=section, agent=agent.name, log_level=log_level)

        store = Path(__file__).parent.parent / "output_dir" / db_name / section / "trajectories"
        store.mkdir(exist_ok=True, parents=True)
        store_json = store / f"{agent.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{Path(gt_json).name}"

        self.map_logger = MapLogger.from_json(store_json)

        MapLogger.info(f"Simulation trajectories will be saved into file {store_json.as_posix()}")

        module = importlib.import_module("src.mllm")

        self.agent = getattr(module, agent.value).__call__()
        self.graph_client = Neo4jClient(db_name)

        """
        map status
        """
        self.start_position = None
        self.last_position = None
        self._distance = None

        self.ground_truth_trajectories = GroundTruthTrajectories.from_dict(
            read_json(gt_json)
        )

        """
        backtrack
        """
        self.backtrack = backtrack
        self.score_container: deque[float]
        self.action_container = deque[int]

        self.backtrack_steps = backtrack_steps
        self.backtrack_threshold = backtrack_threshold
        if self.backtrack:
            MapLogger.info(f"Backtrack mechanism equipped with"
                           f" - backtrack steps: {backtrack_steps}"
                           f" - backtrack threshold: {backtrack_threshold}")

            self.score_container = deque(maxlen=backtrack_steps)
            self.action_container = deque(maxlen=backtrack_steps)

    @classmethod
    def from_json(cls,
                  db_name: str,
                  gt_json: str,
                  agent: MultiModels,
                  backtrack: bool = False,
                  backtrack_steps: int = 0,
                  backtrack_threshold: float = 0.75,
                  ) -> Map:
        return cls(
            db_name=db_name,
            gt_json=gt_json,
            agent=agent,
            backtrack=backtrack,
            backtrack_steps=backtrack_steps,
            backtrack_threshold=backtrack_threshold
        )

    def _step(self) -> bool:
        flag = False
        self.last_position = ViewPointPosition.from_dict(self.start_position.to_dict())
        self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
            current_viewpoint=self.viewpoint,
            azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
            
        )
        MapLogger.info("step forward.")
        # store score of each point
        if self.backtrack:
            self.score_container.append(self.update_viewpoint.score)
            self.action_container.append(self.update_viewpoint.pred_action)
            flag = self._should_backtrack()

        return flag

    def _run_for_single_epoch(self,
                              single_traj_gt: GroundTruthTrajectories.SingleTrajectory,
                              **kwargs
                              ) -> None:
        start_time = datetime.now()
        flag = False  # stand for achieve the goal.
        self.last_position: ViewPointPosition
        self.start_position: ViewPointPosition = single_traj_gt.complete_route[0]

        MapLogger.opt(colors=True).info(
            f"The question {single_traj_gt.question_idx} is <red>**{single_traj_gt.question}**</red>")
        MapLogger.opt(colors=True).info(f"The type of this question belongs to <red>**{single_traj_gt.service}**</red>")

        MapLogger.info(f"Starting from {self.start_position}")

        self.map_logger.insert_step(self.start_position, 0)

        with tqdm(total=kwargs["max_steps"], desc='Searching...', unit='step', colour="#a5d8ff") as pbar:
            should_backtrack = False
            for _ in range(kwargs["max_steps"]):
                MapLogger.opt(colors=True).info(f"<blue>{'-' * 100}</blue>")
                self.viewpoint = self.graph_client.retrieve_viewpoint_from_filename(
                    self.start_position.filename
                )

                self.update_viewpoint = self.agent.observe_and_think(
                    question=single_traj_gt.question,
                    viewpoint=self.viewpoint,
                    backtracked=should_backtrack,
                    pred_action_on_start=self.action_container[0] if should_backtrack else None,
                    last_position=self.last_position,
                    curr_position=self.start_position
                )
                MapLogger.info(f"""
                
                Overall Observation: {self.update_viewpoint.observations}
                Perspective Observation: {self.update_viewpoint.perspective_observation}
                Thoughts: {self.update_viewpoint.thought}
                Action: {self.update_viewpoint.pred_action}
                Score: {self.update_viewpoint.score}
                """)

                # update the node attribution to neo4j
                self.graph_client.update_node_attribution(self.update_viewpoint)

                if self.update_viewpoint.pred_action == StopStatus.REACHED.value:
                    flag = True
                    break

                # step to next position
                should_backtrack = self._step()
                self.map_logger.insert_step(
                    self.start_position,
                    self._distance
                )
                pbar.update(1)

                # check if backtrack
                if should_backtrack:
                    self._backtrack()
                    pbar.update(self.backtrack_steps)
            MapLogger.opt(colors=True).info(f"<blue>{'-' * 100}</blue>")

        cost = (datetime.now() - start_time).total_seconds()
        total_weight = sum(self.map_logger.distance_container)
        total_steps = len(self.map_logger.trajectory.queue)

        if flag:
            MapLogger.success(
                f"The agent has reached the target with total_steps {total_steps} total_weight {total_weight}.")
        else:
            MapLogger.warning(
                f"The agent DOES NOT reach the target with total_steps {total_steps} total_weight {total_weight}")

        # set all visited viewpoint in this epoch to history visited
        self.graph_client.set_history_visited()
        self.map_logger.make_single_trajectory(
            question=single_traj_gt.question,
            question_idx=single_traj_gt.question_idx,
            idx=single_traj_gt.idx,
            _from=single_traj_gt._from,  # noqa
            to=self.map_logger.trajectory.queue[-1].filename,
            service=single_traj_gt.service,
            total_weight=total_weight,
            total_steps=total_steps,
            flag=flag,
            cost=cost
        )

    def _run_loop(self, **kwargs):
        for idx, gt in enumerate(self.ground_truth_trajectories.data):
            MapLogger.info(f"Start running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
            MapLogger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")
            self._run_for_single_epoch(gt, **kwargs)
            MapLogger.success(f"Finished running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
            MapLogger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")

            # return  # TODO means only run first trajectory, should be removed after well test.

    def _should_backtrack(self) -> bool:
        flag = False

        if len(self.score_container) == self.backtrack_steps:
            avg_score = np.average(self.score_container)
            if avg_score < self.backtrack_threshold:
                flag = True
                MapLogger.warning(
                    f"Average score {avg_score} with {self.backtrack_steps} is below threshold {self.backtrack_threshold}, backtrack!")
        return flag

    def _backtrack(self):

        backtrack_nodes: List[ViewPointPosition] = list(self.map_logger.trajectory.queue)[-self.backtrack_steps:]
        backtrack_distances: List[float] = self.map_logger.distance_container[-self.backtrack_steps:]
        MapLogger.info(f"Starting backtrack with {self.backtrack_steps}")

        # back and reset start position
        for node, distance in zip(reversed(backtrack_nodes), reversed(backtrack_distances)):
            self.map_logger.insert_step(
                node,
                distance
            )
            self.start_position = node

        MapLogger.success(f"Backtrack to {self.start_position}")

    def run(self, max_steps: int):
        MapLogger.info(f"Start running with {max_steps} steps.")
        self._run_loop(max_steps=max_steps)
        MapLogger.success("Finished running.")

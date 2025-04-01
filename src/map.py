from __future__ import annotations

import importlib
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import deque
from typing import List, Optional, Literal, Union

from utils.client import Neo4jClient
from utils.map_logger import logger
from utils.operation import read_json, check_env_variables, is_increasing, Compass
from utils.items import GroundTruthTrajectories, ViewPointPosition, MultiModels, StopStatus, \
    ViewPointPositionWithObservation, LastStepMemory, ViewPointAttrToUpdate, Direction


class Map:

    @check_env_variables()
    def __init__(self,
                 db_name: str,
                 gt_json: str,
                 agent: MultiModels,
                 backtrack: bool = False,
                 backtrack_steps: int = 0,
                 backtrack_threshold: Optional[float] = 0.75,
                 backtrack_mechanism: Literal["confidence", "topo_distance"] = "confidence",
                 use_backtrack_prompt: bool = False,
                 retrieve: bool = False,
                 retrieve_epoch: int = 3,
                 retrieve_method: Literal["topology", "spatial"] = "topology",
                 retrieve_distance: Union[int, float] = 1,
                 use_history_trajectory: bool = False,
                 history_steps: int = 3
                 ):

        module = importlib.import_module("src.mllm")

        self.agent = getattr(module, agent.value).__call__()
        self.graph_client = Neo4jClient(db_name)

        """
        map status
        """
        self.start_position = None
        self.last_position = None
        self._distance = 0

        self.ground_truth_trajectories: GroundTruthTrajectories = GroundTruthTrajectories.from_dict(
            read_json(gt_json)
        )

        """
        backtrack
        """
        self.backtrack = backtrack

        self.backtrack_steps = backtrack_steps
        self.backtrack_threshold = backtrack_threshold
        self.backtrack_mechanism = backtrack_mechanism

        self.already_backtracked = False
        self.use_backtrack_prompt = use_backtrack_prompt

        if self.backtrack:
            if backtrack_mechanism == "confidence":
                logger.info("Backtrack mechanism equipped with confidence mechanism.")
                logger.info(f"Backtrack mechanism equipped with"
                            f" - backtrack steps: {backtrack_steps}"
                            f" - backtrack threshold: {backtrack_threshold}")
            elif backtrack_mechanism == "topo_distance":
                logger.info("Backtrack mechanism equipped with topology distance mechanism.")
                logger.info(f"Backtrack mechanism equipped with"
                            f" - backtrack steps: {backtrack_steps}")
            else:
                raise ValueError("backtrack_mechanism should be either 'confidence' or 'topo_distance'.")

            self.score_container = deque(maxlen=backtrack_steps)
            self.action_container = deque(maxlen=backtrack_steps)

        """
        retrieve
        """
        self.retrieve = retrieve
        self.retrieve_epoch = retrieve_epoch
        self.retrieve_method = retrieve_method
        self.retrieve_distance = retrieve_distance

        """
        last step memory
        """
        self.last_step_memory: LastStepMemory = LastStepMemory()
        self.next_action_direction: str = Direction.FRONT.value

        """
        use history trajectory
        """
        self.use_history_trajectory = use_history_trajectory
        self.history_steps = history_steps

    @classmethod
    def from_json(cls,
                  db_name: str,
                  gt_json: str,
                  agent: MultiModels,
                  backtrack: bool = False,
                  backtrack_steps: int = 0,
                  backtrack_threshold: Optional[float] = 0.75,
                  backtrack_mechanism: Literal["confidence", "topo_distance"] = "confidence",
                  use_backtrack_prompt: bool = False,
                  retrieve: bool = False,
                  retrieve_epoch: int = 3,
                  retrieve_method: Literal["topology", "spatial"] = "topology",
                  retrieve_distance: Union[int, float] = 1,
                  use_history_trajectory: bool = False,
                  history_steps: int = 3
                  ) -> Map:
        return cls(
            db_name=db_name,
            gt_json=gt_json,
            agent=agent,
            backtrack=backtrack,
            backtrack_steps=backtrack_steps,
            backtrack_threshold=backtrack_threshold,
            backtrack_mechanism=backtrack_mechanism,
            use_backtrack_prompt=use_backtrack_prompt,
            retrieve=retrieve,
            retrieve_epoch=retrieve_epoch,
            retrieve_method=retrieve_method,
            retrieve_distance=retrieve_distance,
            use_history_trajectory=use_history_trajectory,
            history_steps=history_steps
        )

    def _step(self) -> bool:
        flag = False
        # set visit count in one node.
        self.graph_client.set_node_visited_once(self.start_position.filename)
        self.last_position = ViewPointPosition.from_dict(self.start_position.to_dict())
        self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
            current_viewpoint=self.viewpoint,
            azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
        )
        # 先求出上一步的方向方位角，才能求出这一步的行走方向
        self.last_step_memory.last_forward_azimuth = Compass.get_step_forward_azimuth(self.last_position,
                                                                                      self.start_position)
        self.next_action_direction = Compass.get_relative_direction(
            self.last_step_memory.last_forward_azimuth,
            self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
        ).value

        logger.info("step forward.")
        # store score of each point
        if self.backtrack:
            if self.backtrack_mechanism == "confidence":
                self.score_container.append(self.update_viewpoint.score)
            elif self.backtrack_mechanism == "topo_distance":
                self.score_container.append(self.graph_client.get_steps_between_two_viewpoints(self.start_position,
                                                                                               self.single_traj_gt.complete_route[
                                                                                                   -1]))
            else:
                raise ValueError("backtrack_mechanism should be either 'confidence' or 'topo_distance'.")
            self.action_container.append(self.update_viewpoint.pred_action)
            flag = self._should_backtrack()

        return flag

    def _run_for_single_epoch(self,
                              single_traj_gt: GroundTruthTrajectories.SingleTrajectory,
                              **kwargs
                              ) -> None:
        self.single_traj_gt = single_traj_gt
        for round_num in range(1, kwargs["repeat_num_for_single_question"] + 1):
            start_time = datetime.now()
            flag = False  # stand for achieve the goal.
            self.last_position: ViewPointPosition
            self.start_position: ViewPointPosition = single_traj_gt.complete_route[0]

            logger.opt(colors=True).info(
                f"The question {single_traj_gt.question_idx} is <red>**{single_traj_gt.question}**</red>")
            logger.opt(colors=True).info(
                f"The type of this question belongs to <red>**{single_traj_gt.service}**</red>")

            logger.info(f"Starting from {self.start_position} | Repeat num: {round_num}")

            with tqdm(total=kwargs["max_steps"], desc='Searching...', unit='step', colour="#a5d8ff") as pbar:

                current_step = 1
                while current_step <= kwargs["max_steps"]:
                    logger.opt(colors=True).info(f"<blue>{'-' * 100}</blue>")
                    self.viewpoint = self.graph_client.retrieve_viewpoint_from_filename(
                        self.start_position.filename
                    )
                    if self.already_backtracked and self.use_backtrack_prompt:
                        prompt_perspective_idx = self.graph_client.get_proper_perspective_after_backtrack(
                            self.start_position,
                            self.ground_truth_trajectories.data[-1].complete_route[-1],
                            current_walkable_headings=self.viewpoint.walkable_headings
                        )
                    else:
                        prompt_perspective_idx = None

                    params = {
                        "question": single_traj_gt.question,
                        "viewpoint": self.viewpoint,
                        "last_position": self.last_position,
                        "curr_position": self.start_position,
                        "last_forward_azimuth": self.last_step_memory.last_forward_azimuth,
                        "prompt_perspective_idx": prompt_perspective_idx,
                        "backtracked": self.already_backtracked
                    }

                    if self.retrieve and round_num % self.retrieve_epoch == 0:
                        logger.info("retrieving...")
                        if self.retrieve_method == "topology":
                            retrieved_info = self.graph_client.query_topology_distance(self.start_position.filename,
                                                                                       topology_distance=self.retrieve_distance)
                        elif self.retrieve_method == "spatial":
                            retrieved_info = self.graph_client.query_spatial_distance(self.start_position.filename,
                                                                                      spatial_distance=self.retrieve_distance)
                        else:
                            raise NotImplementedError(
                                f"{self.retrieve_method} is not implemented. Expected 'spatial' or 'topology'")

                        params.update({
                            "retrieved_information": retrieved_info
                        })

                    if self.use_history_trajectory and current_step > self.history_steps:
                        logger.info("using history trajectories...")
                        history_nodes = self.graph_client.get_serval_nodes(
                            list(logger.trajectory.queue)[-self.history_steps:])  # 顺序

                        params.update({
                            "history_nodes_prompt": history_nodes
                        })

                    self.update_viewpoint: ViewPointAttrToUpdate = self.agent.observe_and_think(
                        **params
                    )
                    self.already_backtracked = False
                    logger.info(f"""
                    
                    Overall Observation: {self.update_viewpoint.observations}
                    Perspective Observation: {self.update_viewpoint.perspective_observation}
                    Thoughts: {self.update_viewpoint.thought}
                    Action: {self.update_viewpoint.pred_action}
                    Score: {self.update_viewpoint.score}
                    """)

                    # update the node attribution to neo4j
                    self.graph_client.update_node_attribution(self.update_viewpoint)


                        # step to next position
                    logger.insert_step(
                        ViewPointPositionWithObservation.from_viewpoint_position(
                            self.start_position,
                            self.update_viewpoint
                        ),
                        self._distance
                    )
                    
                    if self.update_viewpoint.pred_action == StopStatus.REACHED.value:
                        flag = True                    
                        break

                    should_backtrack = self._step()

                    # update the extra node attribution in neo4j
                    self.graph_client.set_node_in_current_round(
                        filename=self.update_viewpoint.filename,
                        current_round=round_num,
                        thought=self.update_viewpoint.thought,
                        last_step_filename=self.last_step_memory.last_step_filename,
                        last_action=self.last_step_memory.last_action,
                        last_action_direction=self.last_step_memory.last_action_direction,
                        last_score=self.last_step_memory.last_score,
                        next_step_filename=self.start_position.filename,
                        next_action=self.update_viewpoint.pred_action,
                        next_action_direction=self.next_action_direction,
                        next_score=self.update_viewpoint.score
                    )

                    # update the edge attribution in neo4j
                    self.graph_client.set_edge_in_current_round(
                        round=round_num,
                        step=current_step,
                        action=self.update_viewpoint.pred_action,
                        action_direction=self.next_action_direction,
                        source_filename=self.last_position.filename,
                        target_filename=self.start_position.filename
                    )

                    self._update_last_step_memory()

                    current_step += 1
                    pbar.update(1)

                    # check if backtrack
                    if should_backtrack:
                        self._backtrack()
                        self.backtrack_steps = True
                        pbar.update(self.backtrack_steps)
                        current_step += self.backtrack_steps

                logger.opt(colors=True).info(f"<blue>{'-' * 100}</blue>")

            cost = (datetime.now() - start_time).total_seconds()
            total_weight = sum(logger.distance_container)
            total_steps = len(logger.trajectory.queue)

            # set flag in this round
            self.graph_client.set_node_round_success(round_num, flag)

            if flag:
                logger.success(
                    f"The agent has reached the target with total_steps {total_steps} total_weight {total_weight}.")
            else:
                logger.warning(
                    f"The agent DOES NOT reach the target with total_steps {total_steps} total_weight {total_weight}")

            # set all visited viewpoint in this epoch to history visited
            self.graph_client.set_history_visited()

            # clean if reached retrieve_epoch
            if round_num % self.retrieve_epoch == 0:
                self.graph_client.reset_node_attribution(self.retrieve_epoch)
                self.graph_client.reset_edge_attribution(self.retrieve_epoch)

            if self.start_position.filename == single_traj_gt.complete_route[-1].filename:
                round_success = True
            else:
                round_success = False

            logger.make_single_trajectory(
                question=single_traj_gt.question,
                question_idx=single_traj_gt.question_idx,
                idx=single_traj_gt.idx,
                _from=single_traj_gt._from,  # noqa
                to=logger.trajectory.queue[-1].filename,
                service=single_traj_gt.service,
                total_weight=total_weight,
                total_steps=total_steps,
                flag=flag,
                cost=cost,
                round_num=round_num,
                round_success=round_success
            )

    def _update_last_step_memory(self):
        self.last_step_memory.last_step_filename = self.update_viewpoint.filename
        self.last_step_memory.last_action = self.update_viewpoint.pred_action
        self.last_step_memory.last_action_direction = self.next_action_direction,
        self.last_step_memory.last_score = self.update_viewpoint.score

    def _run_loop(self, **kwargs):
        for idx, gt in enumerate(self.ground_truth_trajectories.data):
            logger.info(f"Start running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
            logger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")
            self._run_for_single_epoch(gt, **kwargs)
            logger.success(f"Finished running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
            logger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")

    def _should_backtrack(self) -> bool:
        flag = False
        if len(self.score_container) == self.backtrack_steps:
            if self.backtrack_mechanism == "confidence":
                avg_score = np.average(self.score_container)
                if avg_score < self.backtrack_threshold:
                    flag = True
                    logger.warning(
                        f"Average score {avg_score} with {self.backtrack_steps} is below threshold {self.backtrack_threshold}, backtrack!")
            elif self.backtrack_mechanism == "topo_distance":
                flag = is_increasing(self.score_container)
                if flag:
                    logger.warning(
                        f"Topology distance {self.score_container} is increasing, backtrack!")
            else:
                logger.error(f"Unknown backtrack mechanism {self.backtrack_mechanism}")

        return flag

    def _backtrack(self):
        """
        backtrack to a new start position
        """
        backtrack_nodes: List[ViewPointPosition] = list(logger.trajectory.queue)[-self.backtrack_steps - 1: -1]
        backtrack_distances: List[float] = logger.distance_container[-self.backtrack_steps - 1: -1]
        logger.info(f"Starting backtrack with {self.backtrack_steps}")

        # back and reset start position
        for node, distance in zip(reversed(backtrack_nodes), reversed(backtrack_distances)):
            logger.insert_step(
                node,
                distance
            )
            self.start_position = node

        logger.success(f"Backtrack to {self.start_position}")

    def run(self, max_steps: int, repeat_num_for_single_question: int = 1):
        logger.info(f"Start running with {max_steps} steps and {repeat_num_for_single_question} repeats.")
        self._run_loop(
            max_steps=max_steps, repeat_num_for_single_question=repeat_num_for_single_question
        )
        logger.success("Finished running.")

from __future__ import annotations

import importlib
import numpy as np

from tqdm import tqdm
from typing import List
from pathlib import Path
from collections import deque

from utils.client import Neo4jClient
from utils.map_logger import MapLogger, logger
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
                 max_retry: int = 3,  # 最大重试次数，改动
                 default_score: float = 0.5,  # 默认评分，改动
                 default_action: int = 0  # 默认动作，改动
                 ):

        self.ground_truth_trajectories = GroundTruthTrajectories.from_dict(
            read_json(gt_json)
        )
        module = importlib.import_module("src.mllm")

        self.agent = getattr(module, agent.value)()
        self.graph_client = Neo4jClient(db_name)

        """
        map status
        """
        self.start_position = None
        self._distance = None
        store = Path(__file__).parent.parent / "logs"
        store_json = store / Path(gt_json).name
        store.mkdir(exist_ok=True)
        self.map_logger = MapLogger.from_json(store_json)

        logger.info(f"Simulation trajectories will be saved into file {store_json.as_posix()}")

        """
        backtrack
        """
        self.backtrack = backtrack
        self.score_container: deque[float]
        self.action_container = deque[int]

        self.backtrack_steps = backtrack_steps
        self.backtrack_threshold = backtrack_threshold
        if self.backtrack:
            logger.info(f"Backtrack mechanism equipped with"
                        f" - backtrack steps: {backtrack_steps}"
                        f" - backtrack threshold: {backtrack_threshold}")

            self.score_container = deque(maxlen=backtrack_steps)
            self.action_container = deque(maxlen=backtrack_steps)

        # 参数化的默认值和重试次数，改动
        self.max_retry = max_retry  # 最大重试次数
        self.default_score = default_score  # 默认评分值
        self.default_action = default_action  # 默认动作索引

    @classmethod
    def from_json(cls,
                  db_name: str,
                  gt_json: str,
                  agent: MultiModels,
                  backtrack: bool,
                  backtrack_steps: int = 0,
                  backtrack_threshold: float = 0.75,
                  ) -> Map:
        return cls(
            db_name=db_name,
            gt_json=gt_json,
            agent=agent.__init__(),
            backtrack=backtrack,
            backtrack_steps=backtrack_steps,
            backtrack_threshold=backtrack_threshold
        )

    # def _step(self) -> bool:
    #     flag = False
    #     self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
    #         current_viewpoint=self.viewpoint,
    #         azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
    #     )
    #     logger.info("step forward.")
    #     # store score of each point
    #     if self.backtrack:
    #         self.score_container.append(self.update_viewpoint.score)
    #         self.action_container.append(self.update_viewpoint.pred_action)
    #         flag = self._should_backtrack()

    #     return flag

    #改动：增加对 pred_action 索引范围的检查，防止超出范围，可能适用于中模型
    # def _step(self) -> bool:
    #     flag = False
    #     try:
    #         self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
    #             current_viewpoint=self.viewpoint,
    #             azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
    #         )
    #         logger.info(f"Step forward successfully with action index: {self.update_viewpoint.pred_action}.")
            
    #     except IndexError:
    #         # 保存原始的action，并处理索引超出范围的情况
    #         original_action = self.update_viewpoint.pred_action
    #         # 调整action到最后一个有效索引
    #         self.update_viewpoint.pred_action = len(self.viewpoint.walkable_headings) - 1
    #         logger.error(f"Invalid action index: {original_action}. Adjusted to last available action: {self.update_viewpoint.pred_action}.")
    #         # 重新计算起始位置和距离
    #         self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
    #             current_viewpoint=self.viewpoint,
    #             azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
    #         )
            
    #     # store score of each point
    #     if self.backtrack:
    #         self.score_container.append(self.update_viewpoint.score)
    #         self.action_container.append(self.update_viewpoint.pred_action)
    #         flag = self._should_backtrack()

    #     return flag
    
    #改动：增加对 pred_action 索引范围的检查，防止超出范围，可能适用于小模型
    # def _step(self) -> bool:
    #     flag = False
    #     max_try = 3  # 设置最大重试次数
    #     for attempt in range(1, max_try + 1):
    #         try:
    #             self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
    #                 current_viewpoint=self.viewpoint,
    #                 azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
    #             )
    #             logger.info(f"Step forward successfully with action index: {self.update_viewpoint.pred_action}.")
    #             break
    #         except IndexError:
    #             # 保存原始的 action，并处理索引超出范围的情况
    #             original_action = self.update_viewpoint.pred_action
    #             self.update_viewpoint.pred_action = len(self.viewpoint.walkable_headings) - 1  # 调整到最后一个有效索引
    #             logger.error(f"Invalid action index: {original_action}. Adjusted to last available action: {self.update_viewpoint.pred_action}.")
    #             if attempt == max_try:
    #                 logger.error(f"Max retries reached in _step. Unable to resolve action index.")
    #         except Exception as e:
    #             logger.error(f"Attempt #{attempt} failed in _step: {e}")
    #             if attempt == max_try:
    #                 raise e  # 重试失败后抛出异常

    #     # 存储评分和动作，检查是否需要回溯
    #     if self.backtrack:
    #         self.score_container.append(self.update_viewpoint.score)
    #         self.action_container.append(self.update_viewpoint.pred_action)
    #         flag = self._should_backtrack()

    #     return flag

    def _step(self) -> bool:
        flag = False
        max_try = self.max_retry  # 设置最大重试次数
        for attempt in range(1, max_try + 1):
            try:
                # 检查 action 是否超出范围
                if self.update_viewpoint.pred_action >= len(self.viewpoint.walkable_headings):
                    original_action = self.update_viewpoint.pred_action
                    self.update_viewpoint.pred_action = len(self.viewpoint.walkable_headings) - 1  # 设置为最后一个有效索引
                    logger.warning(f"Action index {original_action} out of range. Adjusted to {self.update_viewpoint.pred_action}.")

                # 获取最接近的视点
                self.start_position, self._distance = self.graph_client.get_closest_viewpoint(
                    current_viewpoint=self.viewpoint,
                    azimuth=self.viewpoint.walkable_headings[self.update_viewpoint.pred_action]
                )
                logger.info(f"Attempt #{attempt}: Successfully moved to next step with action index {self.update_viewpoint.pred_action}.")
                break
            except IndexError:
                logger.warning(f"Attempt #{attempt}: Invalid action index {self.update_viewpoint.pred_action}. Retrying...")
            except Exception as e:
                logger.error(f"Attempt #{attempt}: Failed with error: {e}")
                if attempt == max_try:
                    logger.error("All attempts failed. Proceeding with default response for step.")
                    self.start_position = self.viewpoint  # 默认值
                    self._distance = 0.0

        # 存储评分和动作，检查是否需要回溯
        if self.backtrack:
            self.score_container.append(self.update_viewpoint.score)
            self.action_container.append(self.update_viewpoint.pred_action)
            flag = self._should_backtrack()

        return flag

#以上

    def _run_for_single_epoch(self,
                              single_traj_gt: GroundTruthTrajectories.SingleTrajectory,
                              **kwargs
                              ) -> None:
        flag = False  # stand for achieve the goal.
        self.start_position: ViewPointPosition = single_traj_gt.complete_route[0]
        logger.opt(colors=True).info(f"The question {single_traj_gt.question_idx} is <red>**{single_traj_gt.question}**</red>")
        logger.opt(colors=True).info(f"The type of this question belongs to <red>**{single_traj_gt.service}**</red>")

        logger.info(f"Starting from {self.start_position}")

        self.map_logger.insert_step(self.start_position, 0)

        with tqdm(total=kwargs["max_steps"], desc='Searching...', unit='step') as pbar:
            should_backtrack = False
            for _ in range(kwargs["max_steps"]):
                logger.opt(colors=True).info(f"<blue>{'-' * 100}</blue>")
                self.viewpoint = self.graph_client.retrieve_viewpoint_from_filename(
                    self.start_position.filename
                )

                # self.update_viewpoint = self.agent.observe_and_think(
                #     question=single_traj_gt.question,
                #     viewpoint=self.viewpoint,
                #     backtracked=should_backtrack,
                #     pred_action_on_start=self.action_container[0] if should_backtrack else None
                # )

                # 改动：调用 observe_and_think 方法，返回结果，可能适用于中模型
                # try:
                #     self.update_viewpoint = self.agent.observe_and_think(
                #         question=single_traj_gt.question,
                #         viewpoint=self.viewpoint,
                #         backtracked=should_backtrack,
                #         pred_action_on_start=self.action_container[0] if should_backtrack else None
                #     )
                # except Exception as e:
                #     logger.error(f"Error during observe_and_think: {e}")
                #     continue  # 跳过当前循环，尝试下一步

                # 可能适用于小模型


                try:
                    max_try = self.max_retry
                    for attempt in range(1, max_try + 1):
                        try:
                            self.update_viewpoint = self.agent.observe_and_think(
                                question=single_traj_gt.question,
                                viewpoint=self.viewpoint,
                                backtracked=should_backtrack,
                                pred_action_on_start=self.action_container[0] if should_backtrack else None
                            )
                            logger.info(f"Attempt #{attempt}: Successfully obtained update_viewpoint.")
                            break
                        except Exception as e:
                            logger.error(f"Attempt #{attempt}: Error during observe_and_think: {e}")
                            if attempt == max_try:
                                logger.error(f"Max retries reached for observe_and_think.")
                                self.update_viewpoint = ViewPointAttrToUpdate(
                                    filename=self.viewpoint.filename,
                                    observations="No observation due to failure",
                                    perspective_observation="No perspective observation",
                                    thought="Default response due to failures",
                                    score=self.default_score,
                                    pred_action=self.default_action,
                                    visited=VisitStatus.CURRENT_VISITED
                                )
                except Exception as e:
                    logger.error(f"Error during observe_and_think after retries: {e}")
                    continue  # 跳过当前循环，尝试下一步
#以上


                logger.info(f"""
                
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
            logger.opt(colors=True).info(f"<blue>{'-' * 100}</blue>")

        total_weight = sum(self.map_logger.distance_container)
        total_steps = len(self.map_logger.trajectory.queue)

        if flag:
            logger.success(f"The agent has reached the target with total_steps {total_steps} total_weight {total_weight}.")
        else:
            logger.warning(f"The agent DOES NOT reach the target with total_steps {total_steps} total_weight {total_weight}")

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
            flag=flag
        )

    # def _run_loop(self, **kwargs):
    #     for idx, gt in enumerate(self.ground_truth_trajectories.data):
    #         logger.info(f"Start running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
    #         logger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")
    #         self._run_for_single_epoch(gt, **kwargs)
    #         logger.success(f"Finished running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
    #         logger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")

    #         return  # TODO means only run first trajectory, should be removed after well test.
    
    #改动：增加了保障机制以捕获 _run_for_single_epoch 中的错误，确保整个循环能够正常运行。
    def _run_loop(self, **kwargs):
        for idx, gt in enumerate(self.ground_truth_trajectories.data):
            logger.info(f"Start running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
            logger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")
            try:
                self._run_for_single_epoch(gt, **kwargs)
            except Exception as e:
                logger.error(f"Error during running trajectory {idx + 1}: {e}")
                continue  # 跳过当前轨迹，尝试下一轨迹
            logger.success(f"Finished running trajectory {idx + 1} / {len(self.ground_truth_trajectories.data)}")
            logger.opt(colors=True).info(f"<blue>{'=' * 112}</blue>")

            return  # TODO means only run first trajectory, should be removed after well test.


    def _should_backtrack(self) -> bool:
        flag = False

        if len(self.score_container) == self.backtrack_steps:
            avg_score = np.average(self.score_container)
            if avg_score < self.backtrack_threshold:
                flag = True
                logger.warning(f"Average score {avg_score} with {self.backtrack_steps} is below threshold {self.backtrack_threshold}, backtrack!")
        return flag

    def _backtrack(self):

        backtrack_nodes: List[ViewPointPosition] = list(self.map_logger.trajectory.queue)[-self.backtrack_steps:]
        backtrack_distances: List[float] = self.map_logger.distance_container[-self.backtrack_steps:]
        logger.info(f"Starting backtrack with {self.backtrack_steps}")

        # back and reset start position
        for node, distance in zip(reversed(backtrack_nodes), reversed(backtrack_distances)):
            self.map_logger.insert_step(
                node,
                distance
            )
            self.start_position = node

        logger.success(f"Backtrack to {self.start_position}")

    def run(self, max_steps: int = 35):
        logger.info(f"Start running with {max_steps} steps.")
        self._run_loop(max_steps=max_steps)
        logger.success("Finished running.")

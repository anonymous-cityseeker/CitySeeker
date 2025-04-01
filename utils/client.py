import os
from typing import List
from langchain_neo4j import Neo4jGraph

from utils.map_logger import logger
from utils.operation import Compass, find_closest_value
from utils.items import ViewPointAttrToUpdate, ViewPoint, VisitStatus, ViewPointPosition, ViewPointPositionWithObservation


class Neo4jClient:

    def __init__(self, db_name: str):
        params = {
            "url": os.getenv("NEO4J_URL"), #"url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": db_name,
        }
        logger.info(f"Connecting to neo4j with database {db_name}...")
        self.client = Neo4jGraph(**params)
        logger.info("Neo4j connected.")

    @classmethod
    def _format_node(cls, node):
        ordered_node = {key: node[key] for key in
                        ["filename", "visited", "observations", "perspective_observation", "thought",
                         "pred_action", "score", "longitude", "latitude", "heading", "walkable_headings"]
                        if key in node}
        ordered_node.setdefault("visited", VisitStatus.UNVISITED.value)  # 设置默认值
        return ordered_node

    @classmethod
    def _parse_node(cls, result):
        nodes = {}
        relationships = []

        for record in result:
            source = record["source_node"]
            target = record["target_node"]
            relationship_list = record["relationship_properties"]

            # 添加日志，查看 source 和 target 的 filename
            logger.info(f"Source node filename: {source.get('filename')}")
            logger.info(f"Target node filename: {target.get('filename')}")


            if not source.get("filename") or not target.get("filename"):
                logger.warning("⚠ Skipping entry due to missing filename.")
                continue

            formatted_source = cls._format_node(source)
            formatted_target = cls._format_node(target)

            # 先存 relationships
            for relationship in relationship_list:
                relationships.append({
                    "source": formatted_source["filename"],
                    "target": formatted_target["filename"],
                    "relationship_properties": relationship  # 保留原始的 relationship 信息
                })

            # 再存 nodes（去重）
            nodes[formatted_source["filename"]] = formatted_source
            nodes[formatted_target["filename"]] = formatted_target
             
            # 打印解析后的 relationships 和 nodes
            logger.info(f"Parsed relationships: {relationships}")
            logger.info(f"Parsed nodes: {list(nodes.values())}")

        return {"relationships": relationships, "nodes": list(nodes.values())}  # ✅ 先返回 relationships，再返回 nodes

    def query_topology_distance(self, filename: str, topology_distance: int = 1):
        """
        查询 `filename` 对应的 `Point`，获取 `n` 跳以内的所有 `CONNECTED_TO` 关系。
        返回 {"relationships": [...], "nodes": [...]}
        """
        query = f"""
        MATCH path = (start:Point {{filename: '{filename}'}})-[r:CONNECTED_TO*1..{topology_distance}]-(other)
        RETURN 
            [rel IN relationships(path) | properties(rel)] AS relationship_properties,
            apoc.map.removeKeys(properties(start), ['gtype', 'bbox', 'month', 'year']) AS source_node,
            apoc.map.removeKeys(properties(other), ['gtype', 'bbox', 'month', 'year']) AS target_node
        """

        result = self.client.query(query)

        # 打印原始查询结果
        if result:
            logger.info(f"Raw Cypher query result: {result}")
        else:
            logger.warning("⚠ No data returned from Neo4j query.")
            return {"relationships": [], "nodes": []}  # 如果没有结果，直接返回空的 relationships 和 nodes


        return self._parse_node(result)

    def query_spatial_distance(self, filename: str, spatial_distance: float = 100.0):
        """
        查询 `filename` 对应的 `Point`，在 `spatial_distance` 范围内的所有 `CONNECTED_TO` 关系。
        返回 {"relationships": [...], "nodes": [...]}
        """
        query = f"""
        MATCH (start:Point {{filename: '{filename}'}})
        CALL spatial.withinDistance("locations", start, {spatial_distance / 1000}) 
        YIELD node AS nearby_node
        MATCH (nearby_node)-[r:CONNECTED_TO]->(other_nearby)
        RETURN 
            properties(r) AS relationship_properties,
            apoc.map.removeKeys(properties(nearby_node), ['gtype', 'bbox', 'month', 'year']) AS source_node,
            apoc.map.removeKeys(properties(other_nearby), ['gtype', 'bbox', 'month', 'year']) AS target_node
        """

        result = self.client.query(query)

        # 打印原始查询结果
        if result:
            logger.info(f"Raw Cypher query result: {result}")
        else:
            logger.warning("⚠ No data returned from Neo4j query.")
            return {"relationships": [], "nodes": []}  # 如果没有结果，直接返回空的 relationships 和 nodes

        return self._parse_node(result)

    def update_node_attribution(self, update_viewpoint: ViewPointAttrToUpdate):
        query = """
        MATCH (n:Point {filename: $filename})
        SET n += $new_properties
        RETURN n
        """

        self.client.query(
            query,
            params={
                "filename": update_viewpoint.filename,
                "new_properties": update_viewpoint.to_dict(encode_json=True)
            }
        )
        logger.info(f"Node {update_viewpoint.filename} attributions updated.")

    def set_node_visited_once(self, filename):
        query = """
        MATCH (n: Point {filename: $filename})
        SET n.total_visits = coalesce(n.total_visits, 0) + 1
        RETURN n.total_visits
        """

        self.client.query(
            query,
            params={
                "filename": filename
            }
        )

    def reset_node_attribution(self, round: int):
        query = """
        MATCH (n: Point)
        SET n.total_visits = 0,
        """ + ",".join(
            [f'n.round_{i} = "{{}}"' for i in range(1, round + 1)]
        ) + ' RETURN n;'

        self.client.query(query)

    def reset_edge_attribution(self, round: int):
        query = """
        MATCH (r: CONNECTED_TO)
        SET
        """ + ",".join(
            [f'r.round_{i} = ""' for i in range(1, round + 1)]
        ) + ' RETURN r;'

        self.client.query(query)



    # def reset_node_attribution(self, round: int):
    #     query = """
    #     MATCH (n: Point)
    #     SET n.total_visits = 0,
    #     """ + ",".join(
    #         [f'n.round_{i} = "{{}}' for i in range(1, round + 1)]
    #     ) + ' RETURN n;'

    #     self.client.query(query)

    # def reset_edge_attribution(self, round: int):
    #     query = """
    #     MATCH (r: CONNECTED_TO)
    #     SET
    #     """ + ",".join(
    #         [f"r.round_{i} = ''" for i in range(1, round + 1)]
    #     ) + ' RETURN r;'
    #     self.client.query(query)

    def set_node_in_current_round(self,
                                  filename: str,
                                  current_round: int,
                                  thought: str,
                                  last_step_filename: str,
                                  last_action: int,
                                  last_action_direction: str,
                                  last_score: float,
                                  next_step_filename: str,
                                  next_action: int,
                                  next_action_direction: str,
                                  next_score: float):
        query = f"""
        MATCH (n: Point {{filename: "{filename}"}})
        SET n.%s = "{{
        thought = {thought},
        round = {current_round},
        last_step_filename = {last_step_filename},
        last_action = {last_action},
        last_action_direction = {last_action_direction},
        last_score = {last_score},
        next_step_filename = {next_step_filename},
        next_action = {next_action},
        next_action_direction = {next_action_direction},
        next_score = {next_score},
        round_success = unknown,
        }}"
        """ % f"round_{current_round}"
        self.client.query(
            query
        )

    def set_edge_in_current_round(self,
                                  round: int,
                                  step: int,
                                  action: int,
                                  action_direction: str,
                                  source_filename: str,
                                  target_filename: str):
        query = f"""
        MATCH (a:Point {{filename: $source_filename}})-[r:CONNECTED_TO]->(b:Point {{filename: $target_filename}})
        SET r.round_{round} = $step,  r.action = $action, r.action_direction = $action_direction
        RETURN r;
        """
        self.client.query(
            query,
            params={
                "source_filename": source_filename,
                "target_filename": target_filename,
                "step": f"step{step}",
                "action": action,
                "action_direction": action_direction,
            }
        )

    def set_node_round_success(self, round: int, flag: bool = True):
        query = f"""
        MATCH (n)
        WHERE n.round_{round} IS NOT NULL
        SET n.round_{round} = apoc.text.replace(n.round_{round}, 'round_success = unknown', 'round_success = {flag}')
        """
        self.client.query(
            query
        )


    def reset_all_viewpoint_after_epoch(self):
        status: dict = ViewPointAttrToUpdate().to_dict(encode_json=True)
        status.pop("filename")

        query = """
        MATCH (n:Point)
        SET n += $default_properties
        RETURN n
        """
        self.client.query(
            query,
            params={"default_properties": status}
        )

    def retrieve_viewpoint_from_filename(self,
                                         filename: str) -> ViewPoint:
        query = """
        MATCH (n:Point {filename: $filename})
        RETURN n
        """
        viewpoint = self.client.query(
            query,
            params={"filename": filename}
        )[0]["n"]

        return ViewPoint(
            filename=filename,
            heading=viewpoint["heading"],
            walkable_headings=viewpoint["walkable_headings"]
        )

    def retrieve_viewpoint_from_element_id(self,
                                           element_id: str) -> ViewPointPosition:
        query = """
        MATCH (n: Point)
        WHERE elementID(n) = $elementID
        RETURN n
        """

        viewpoint = self.client.query(query, params={"elementID": element_id})[0]["n"]

        return ViewPointPosition(
            filename=viewpoint["filename"],
            longitude=viewpoint["longitude"],
            latitude=viewpoint["latitude"]
        )

    def retrieve_edges_start_from_viewpoint(self,
                                            filename: str) -> List:
        query = """
        MATCH (n:Point {filename: $filename})-[r:CONNECTED_TO]->(m)
        RETURN elementID(n) AS startNodeId, elementID(m) AS endNodeId, properties(r) AS rProperties
        """

        relationships: list = self.client.query(query, params={"filename": filename})

        return relationships

    def get_closest_viewpoint(self,
                              current_viewpoint: ViewPoint,
                              azimuth: float) -> (ViewPointPosition, float):
        edges = self.retrieve_edges_start_from_viewpoint(current_viewpoint.filename)
        closest = min(edges, key=lambda x: abs(x['rProperties']['azimuth'] - azimuth))

        node_id = closest["endNodeId"]

        return self.retrieve_viewpoint_from_element_id(node_id), closest["rProperties"]["distance"]

    def set_history_visited(self):
        query = f"""
        MATCH (n:Point)
        WHERE n.visited is NOT NULL AND n.visited = "{VisitStatus.CURRENT_VISITED.value}"
        SET n.visited = "{VisitStatus.HISTORY_VISITED.value}"
        RETURN n
        """
        self.client.query(query)
        logger.info("All position visited in current epoch has been set to HISTORY_VISITED")

    def get_steps_between_two_viewpoints(self,
                                         start_viewpoint: ViewPointPosition,
                                         end_viewpoint: ViewPointPosition) -> int:
        
        if start_viewpoint.filename == end_viewpoint.filename:
            return 0
        else:
            query = """
            MATCH (a {filename: $start_filename}), (b {filename: $end_filename})
            MATCH p = shortestPath((a)-[:CONNECTED_TO*]-(b))
            RETURN length(p) AS steps
            """

            steps = self.client.query(query, params={"start_filename": start_viewpoint.filename,
                                                    "end_filename": end_viewpoint.filename})
            return steps[0]["steps"]

    def get_proper_perspective_after_backtrack(self,
                                               back_viewpoint: ViewPointPosition,
                                               gt_viewpoint: ViewPointPosition,
                                               current_walkable_headings: list) -> int:
        query = """
        MATCH (a {filename: $start_filename}), (b {filename: $end_filename})
        MATCH p = shortestPath((a)-[:CONNECTED_TO*]-(b))
        RETURN p
        """

        traj = self.client.query(query, params={"start_filename": back_viewpoint.filename,
                                                "end_filename": gt_viewpoint.filename})
        gt_next_viewpoint = traj[0]["p"][2]
        forward_azimuth = Compass.get_step_forward_azimuth(
            last_position=back_viewpoint,
            curr_position=ViewPointPosition(filename=gt_next_viewpoint["filename"],
                                            longitude=gt_next_viewpoint["longitude"], latitude=gt_next_viewpoint["latitude"]),
        )
        idx = find_closest_value(target=forward_azimuth, lst=current_walkable_headings)

        return idx

    # def get_serval_nodes(self, filenames: list) -> List[dict]:
    #     query = """
    #     MATCH (n:Point)
    #     WHERE n.filename IN $filenames
    #     RETURN properties(n);
    #     """

    #     nodes = self.client.query(
    #         query,
    #         params={"filenames": filenames}
    #     )
    #     for idx, node in enumerate(nodes):
    #         node = node["properties(n)"]
    #         for key in list(node.keys()):
    #             if key.startswith("round") or key in ["gtype", "bbox"]:
    #                 node.pop(key)
    #         nodes[idx] = node

    #     return nodes


    def get_serval_nodes(self, trajectories: List[ViewPointPositionWithObservation]) -> List[dict]:
        query = """
        MATCH (n:Point)
        WHERE n.filename IN $filenames
        RETURN properties(n);
        """

        nodes = self.client.query(
            query,
            params={"filenames": [trajectory.filename for trajectory in trajectories]}
        )
        for idx, node in enumerate(nodes):
            node = node["properties(n)"]
            for key in list(node.keys()):
                if key.startswith("round") or key in ["gtype", "bbox"]:
                    node.pop(key)
            nodes[idx] = node

        return nodes
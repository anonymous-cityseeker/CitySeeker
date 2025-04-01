import os
from typing import List
from langchain_neo4j import Neo4jGraph

from utils.map_logger import MapLogger
from utils.items import ViewPointAttrToUpdate, ViewPoint, VisitStatus, ViewPointPosition


class Neo4jClient:

    def __init__(self, db_name: str):
        params = {
            "url": os.getenv("NEO4J_URL"),
            "username": "neo4j",
            "password": os.getenv("NEO4J_PASSWORD"),
            "database": db_name,
        }
        MapLogger.info(f"Connecting to neo4j with database {db_name} and URL: {params['url']}...")
        self.client = Neo4jGraph(**params)
        MapLogger.info("Neo4j connected.")

    def query_topology_distance(self, filename: str, topology_distance: int = 3):
        query = f"""
        MATCH (start:Point {{filename: "{filename}"}})-[:CONNECTED_TO*1..{topology_distance}]-(other)
        WHERE other <> start AND other.visited = true
        RETURN other
        """
        return self.client.query(query)

    def query_spatial_distance(self, filename: str, spatial_distance: float = 100.0):
        query = f"""
        MATCH (start:Point {{filename: '{filename}'}})
        CALL spatial.withinDistance("locations", start, {spatial_distance / 1000}) 
        YIELD node AS other
        WHERE other <> start AND other.visited = true
        RETURN other
        """

        return self.client.query(query)

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
        MapLogger.info(f"Node {update_viewpoint.filename} attributions updated.")

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
        WHERE n.visited is NOT NULL AND n.visited = {VisitStatus.CURRENT_VISITED.value}
        SET n.visited = {VisitStatus.HISTORY_VISITED.value}
        RETURN n
        """
        self.client.query(query)
        MapLogger.info("All position visited in current epoch has been set to HISTORY_VISITED")

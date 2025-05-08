from neo4j import GraphDatabase


class Orchestrator:
    def __init__(self, neo4j_url, neo4j_username, neo4j_pw):
        self.neo4j_url = neo4j_url
        self.neo4j_username = neo4j_username
        self.neo4j_pw = neo4j_pw

    def create_index(self):
        driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_username, self.neo4j_pw))
        with driver.session() as session:
            session.run("CREATE VECTOR INDEX textEmbedding IF NOT EXISTS FOR (n:Document) ON n.embedding")
        driver.close()

    def clear_db(self):
        driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_username, self.neo4j_pw))
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            session.run("DROP INDEX textEmbedding")
        driver.close()

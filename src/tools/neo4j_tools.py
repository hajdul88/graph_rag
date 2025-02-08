from neo4j import GraphDatabase


def create_index(neo4j_url, neo4j_username, neo4j_pw):
    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_username, neo4j_pw))
    with driver.session() as session:
        session.run("CREATE VECTOR INDEX textEmbedding IF NOT EXISTS FOR (n:Chunk) ON n.embedding")
    driver.close()

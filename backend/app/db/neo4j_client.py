import logging
from typing import Dict, Any, List, Optional, Tuple
from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError
import asyncio
from concurrent.futures import ThreadPoolExecutor

from backend.app.config import settings

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        """Initialize Neo4j client"""
        self.uri = uri or settings.neo4j_url
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self.driver: Optional[Driver] = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized Neo4j client: {self.uri}")
    
    async def connect(self):
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            await self._run_async(lambda tx: tx.run("RETURN 1").single())
            logger.info("Connected to Neo4j successfully")
            
            # Initialize schema
            await self._initialize_schema()
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Neo4j"""
        if self.driver:
            self.driver.close()
            logger.info("Disconnected from Neo4j")
    
    async def _run_async(self, work_function):
        """Run Neo4j transaction asynchronously"""
        loop = asyncio.get_event_loop()
        
        def run_transaction():
            with self.driver.session() as session:
                return session.execute_write(work_function)
        
        return await loop.run_in_executor(self.executor, run_transaction)
    
    async def _read_async(self, work_function):
        """Run Neo4j read transaction asynchronously"""
        loop = asyncio.get_event_loop()
        
        def run_transaction():
            with self.driver.session() as session:
                return session.execute_read(work_function)
        
        return await loop.run_in_executor(self.executor, run_transaction)
    
    async def _initialize_schema(self):
        """Initialize Neo4j schema with constraints and indexes"""
        try:
            schema_queries = [
                # Character node constraints
                "CREATE CONSTRAINT character_id_unique IF NOT EXISTS FOR (c:Character) REQUIRE c.character_id IS UNIQUE",
                "CREATE CONSTRAINT character_project_id IF NOT EXISTS FOR (c:Character) REQUIRE c.project_id IS NOT NULL",
                
                # Location node constraints  
                "CREATE CONSTRAINT location_id_unique IF NOT EXISTS FOR (l:Location) REQUIRE l.location_id IS UNIQUE",
                
                # Plot node constraints
                "CREATE CONSTRAINT plot_arc_id_unique IF NOT EXISTS FOR (p:PlotArc) REQUIRE p.arc_id IS UNIQUE",
                
                # Indexes for performance
                "CREATE INDEX character_name_index IF NOT EXISTS FOR (c:Character) ON (c.name)",
                "CREATE INDEX character_project_index IF NOT EXISTS FOR (c:Character) ON (c.project_id)",
                "CREATE INDEX location_name_index IF NOT EXISTS FOR (l:Location) ON (l.name)"
            ]
            
            for query in schema_queries:
                await self._run_async(lambda tx, q=query: tx.run(q))
            
            logger.info("Neo4j schema initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j schema: {e}")
    
    # Character Management
    async def create_character(self, project_id: str, character_data: Dict[str, Any]) -> str:
        """Create a new character node"""
        def create_character_tx(tx):
            query = """
            CREATE (c:Character {
                character_id: $character_id,
                project_id: $project_id,
                name: $name,
                description: $description,
                traits: $traits,
                role: $role,
                created_at: datetime()
            })
            RETURN c.character_id as character_id
            """
            
            result = tx.run(query, 
                character_id=character_data["character_id"],
                project_id=project_id,
                name=character_data["name"],
                description=character_data.get("description", ""),
                traits=character_data.get("traits", []),
                role=character_data.get("role", "supporting")
            )
            return result.single()["character_id"]
        
        character_id = await self._run_async(create_character_tx)
        logger.info(f"Created character: {character_id}")
        return character_id
    
    async def get_character(self, project_id: str, character_id: str) -> Optional[Dict[str, Any]]:
        """Get character by ID"""
        def get_character_tx(tx):
            query = """
            MATCH (c:Character {character_id: $character_id, project_id: $project_id})
            RETURN c {
                .character_id,
                .name,
                .description,
                .traits,
                .role,
                .created_at
            } as character
            """
            
            result = tx.run(query, character_id=character_id, project_id=project_id)
            record = result.single()
            return record["character"] if record else None
        
        return await self._read_async(get_character_tx)
    
    async def create_relationship(self, 
                                project_id: str,
                                char1_id: str, 
                                char2_id: str, 
                                relationship_type: str,
                                properties: Dict[str, Any] = None) -> bool:
        """Create relationship between characters"""
        def create_relationship_tx(tx):
            query = f"""
            MATCH (c1:Character {{character_id: $char1_id, project_id: $project_id}})
            MATCH (c2:Character {{character_id: $char2_id, project_id: $project_id}})
            CREATE (c1)-[r:{relationship_type} $properties]->(c2)
            RETURN r
            """
            
            result = tx.run(query,
                char1_id=char1_id,
                char2_id=char2_id,
                project_id=project_id,
                properties=properties or {}
            )
            return result.single() is not None
        
        success = await self._run_async(create_relationship_tx)
        logger.info(f"Created relationship: {char1_id} -{relationship_type}-> {char2_id}")
        return success
    
    async def get_character_relationships(self, project_id: str, character_id: str) -> List[Dict[str, Any]]:
        """Get all relationships for a character"""
        def get_relationships_tx(tx):
            query = """
            MATCH (c1:Character {character_id: $character_id, project_id: $project_id})
            MATCH (c1)-[r]-(c2:Character)
            RETURN 
                type(r) as relationship_type,
                c2.character_id as related_character_id,
                c2.name as related_character_name,
                properties(r) as relationship_properties,
                startNode(r) = c1 as is_outgoing
            """
            
            result = tx.run(query, character_id=character_id, project_id=project_id)
            return [dict(record) for record in result]
        
        return await self._read_async(get_relationships_tx)
    
    async def get_character_context(self, project_id: str, character_ids: List[str]) -> Dict[str, Any]:
        """Get comprehensive character context for story generation"""
        def get_context_tx(tx):
            query = """
            MATCH (c:Character {project_id: $project_id})
            WHERE c.character_id IN $character_ids
            OPTIONAL MATCH (c)-[r]-(related:Character)
            RETURN 
                c {
                    .character_id,
                    .name,
                    .description,
                    .traits,
                    .role
                } as character,
                collect({
                    related_character: related.name,
                    relationship_type: type(r),
                    properties: properties(r)
                }) as relationships
            """
            
            result = tx.run(query, project_id=project_id, character_ids=character_ids)
            
            context = {
                "characters": [],
                "relationship_network": []
            }
            
            for record in result:
                character = record["character"]
                relationships = record["relationships"]
                
                context["characters"].append(character)
                
                for rel in relationships:
                    if rel["related_character"]:  # Filter out null relationships
                        context["relationship_network"].append({
                            "character": character["name"],
                            "related_to": rel["related_character"],
                            "relationship": rel["relationship_type"],
                            "details": rel["properties"]
                        })
            
            return context
        
        return await self._read_async(get_context_tx)
    
    # Location Management
    async def create_location(self, project_id: str, location_data: Dict[str, Any]) -> str:
        """Create a new location node"""
        def create_location_tx(tx):
            query = """
            CREATE (l:Location {
                location_id: $location_id,
                project_id: $project_id,
                name: $name,
                description: $description,
                location_type: $location_type,
                created_at: datetime()
            })
            RETURN l.location_id as location_id
            """
            
            result = tx.run(query,
                location_id=location_data["location_id"],
                project_id=project_id,
                name=location_data["name"],
                description=location_data.get("description", ""),
                location_type=location_data.get("location_type", "general")
            )
            return result.single()["location_id"]
        
        location_id = await self._run_async(create_location_tx)
        logger.info(f"Created location: {location_id}")
        return location_id
    
    async def connect_character_to_location(self, 
                                          project_id: str,
                                          character_id: str, 
                                          location_id: str,
                                          relationship_type: str = "VISITS") -> bool:
        """Connect character to location"""
        def connect_tx(tx):
            query = f"""
            MATCH (c:Character {{character_id: $character_id, project_id: $project_id}})
            MATCH (l:Location {{location_id: $location_id, project_id: $project_id}})
            CREATE (c)-[r:{relationship_type}]->(l)
            RETURN r
            """
            
            result = tx.run(query,
                character_id=character_id,
                location_id=location_id,
                project_id=project_id
            )
            return result.single() is not None
        
        return await self._run_async(connect_tx)
    
    # Utility Methods
    async def get_project_characters(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all characters in a project"""
        def get_characters_tx(tx):
            query = """
            MATCH (c:Character {project_id: $project_id})
            RETURN c {
                .character_id,
                .name,
                .description,
                .traits,
                .role
            } as character
            ORDER BY c.name
            """
            
            result = tx.run(query, project_id=project_id)
            return [record["character"] for record in result]
        
        return await self._read_async(get_characters_tx)
    
    async def delete_project_data(self, project_id: str) -> bool:
        """Delete all data for a project"""
        def delete_project_tx(tx):
            query = """
            MATCH (n {project_id: $project_id})
            DETACH DELETE n
            RETURN count(n) as deleted_count
            """
            
            result = tx.run(query, project_id=project_id)
            deleted_count = result.single()["deleted_count"]
            return deleted_count > 0
        
        success = await self._run_async(delete_project_tx)
        logger.info(f"Deleted project data: {project_id}")
        return success

# Global Neo4j client instance
neo4j_client = Neo4jClient()

async def init_neo4j():
    """Initialize Neo4j connection"""
    await neo4j_client.connect()

async def get_neo4j() -> Neo4jClient:
    """Get Neo4j client instance"""
    return neo4j_client
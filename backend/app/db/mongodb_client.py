import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from bson import ObjectId

from backend.app.config import settings

logger = logging.getLogger(__name__)

class MongoDBClient:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
    
    async def connect(self):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_url)
            self.db = self.client[settings.mongodb_database]
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info("Connected to MongoDB successfully")
            
            # Create indexes
            await self._create_indexes()
            
        except ServerSelectionTimeoutError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self):
        """Create necessary indexes"""
        try:
            # Jobs collection indexes
            await self.db.jobs.create_index("job_id", unique=True)
            await self.db.jobs.create_index("user_id")
            await self.db.jobs.create_index("project_id")
            await self.db.jobs.create_index("state")
            await self.db.jobs.create_index("created_at")
            
            # Chapter versions collection indexes
            await self.db.chapter_versions.create_index([("project_id", 1), ("chapter_id", 1)])
            await self.db.chapter_versions.create_index("version_number")
            await self.db.chapter_versions.create_index("created_at")
            
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
    
    # Job operations
    async def create_job(self, job_data: Dict[str, Any]) -> str:
        """Create a new job"""
        job_doc = {
            **job_data,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc)
        }
        
        result = await self.db.jobs.insert_one(job_doc)
        return str(result.inserted_id)
    
    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        job = await self.db.jobs.find_one({"job_id": job_id})
        if job:
            job["_id"] = str(job["_id"])
        return job
    
    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """Update job"""
        updates["updated_at"] = datetime.now(timezone.utc)
        result = await self.db.jobs.update_one(
            {"job_id": job_id},
            {"$set": updates}
        )
        return result.modified_count > 0
    
    async def get_jobs_by_user(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get jobs by user ID"""
        cursor = self.db.jobs.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        jobs = []
        async for job in cursor:
            job["_id"] = str(job["_id"])
            jobs.append(job)
        return jobs
    
    # Chapter version operations
    async def create_chapter_version(self, version_data: Dict[str, Any]) -> str:
        """Create a new chapter version"""
        version_doc = {
            **version_data,
            "created_at": datetime.now(timezone.utc)
        }
        
        result = await self.db.chapter_versions.insert_one(version_doc)
        return str(result.inserted_id)
    
    async def get_chapter_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get chapter version by ID"""
        try:
            version = await self.db.chapter_versions.find_one({"_id": ObjectId(version_id)})
            if version:
                version["_id"] = str(version["_id"])
            return version
        except Exception:
            return None
    
    async def get_latest_chapter_version(self, project_id: str, chapter_id: str) -> Optional[Dict[str, Any]]:
        """Get latest version of a chapter"""
        version = await self.db.chapter_versions.find_one(
            {"project_id": project_id, "chapter_id": chapter_id},
            sort=[("version_number", -1)]
        )
        if version:
            version["_id"] = str(version["_id"])
        return version
    
    async def get_chapter_versions(self, project_id: str, chapter_id: str) -> List[Dict[str, Any]]:
        """Get all versions of a chapter"""
        cursor = self.db.chapter_versions.find(
            {"project_id": project_id, "chapter_id": chapter_id}
        ).sort("version_number", -1)
        
        versions = []
        async for version in cursor:
            version["_id"] = str(version["_id"])
            versions.append(version)
        return versions

# Global MongoDB client instance
mongodb_client = MongoDBClient()

async def init_mongodb():
    """Initialize MongoDB connection"""
    await mongodb_client.connect()

async def get_mongodb() -> MongoDBClient:
    """Get MongoDB client instance"""
    return mongodb_client
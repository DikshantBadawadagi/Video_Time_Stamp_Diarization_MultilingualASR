# app/models.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Any, Dict

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/whisper_app")
DB_NAME = os.getenv("MONGODB_DB", "whisper_app")

class MongoDB:
    def __init__(self, uri: str = MONGODB_URI):
        self.client: AsyncIOMotorClient | None = None
        self.uri = uri
        self.db = None

    async def connect(self):
        if not self.client:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[DB_NAME]

    async def close(self):
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    async def insert_analysis(self, collection: str, document: Dict[str, Any]):
        await self.connect()
        coll = self.db[collection]
        res = await coll.insert_one(document)
        return str(res.inserted_id)

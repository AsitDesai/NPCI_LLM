#!/usr/bin/env python3
"""
Clear Qdrant Database Script

This script clears all existing embeddings and collections from the Qdrant database
to start fresh with the new simplified architecture.
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient
from config.settings import settings
from config.logging import get_logger

logger = get_logger(__name__)


def clear_qdrant_database():
    """Clear all collections and data from Qdrant database."""
    print("🧹 CLEARING QDRANT DATABASE")
    print("="*50)
    
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=False
        )
        
        print(f"📡 Connected to Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")
        
        # Get all collections
        collections = client.get_collections()
        existing_collections = [col.name for col in collections.collections]
        
        if not existing_collections:
            print("✅ No collections found - database is already clean")
            return True
        
        print(f"📋 Found {len(existing_collections)} collections:")
        for collection in existing_collections:
            print(f"   - {collection}")
        
        # Delete each collection
        print(f"\n🗑️ Deleting collections...")
        for collection_name in existing_collections:
            try:
                client.delete_collection(collection_name)
                print(f"   ✅ Deleted collection: {collection_name}")
            except Exception as e:
                print(f"   ❌ Failed to delete collection {collection_name}: {e}")
        
        # Verify deletion
        collections_after = client.get_collections()
        remaining_collections = [col.name for col in collections_after.collections]
        
        if not remaining_collections:
            print(f"\n✅ Successfully cleared all collections!")
            print(f"   Database is now clean and ready for new data")
            return True
        else:
            print(f"\n❌ Some collections could not be deleted:")
            for collection in remaining_collections:
                print(f"   - {collection}")
            return False
            
    except Exception as e:
        logger.error(f"Error clearing Qdrant database: {e}")
        print(f"\n❌ FAILED: {e}")
        return False


def main():
    """Main function to clear the database."""
    print("⚠️  WARNING: This will delete ALL data from Qdrant database!")
    print("   This action cannot be undone.")
    print()
    
    # Ask for confirmation
    response = input("Do you want to continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        success = clear_qdrant_database()
        if success:
            print(f"\n🎉 Database cleared successfully!")
            print(f"   Ready for new simplified architecture")
        else:
            print(f"\n❌ Failed to clear database completely")
            return False
    else:
        print(f"\n❌ Operation cancelled by user")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

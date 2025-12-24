"""
Example: Deleting Notebooks with the Arato Python SDK

This example demonstrates how to use the new delete notebook functionality.
"""

from arato_client import AratoClient

# Initialize the client with your API key
client = AratoClient(api_key="your-api-key-here")

# Example 1: Create and delete a notebook
print("Creating a test notebook...")
notebook = client.notebooks.create(
    name="Test Notebook to Delete",
    description="This notebook will be deleted"
)
notebook_id = notebook["id"]
print(f"Created notebook: {notebook_id}")

# Delete the notebook
print(f"Deleting notebook {notebook_id}...")
client.notebooks.delete(notebook_id)
print("Notebook deleted successfully!")

# Example 2: Handling errors when deleting
try:
    # Try to delete a non-existent notebook
    client.notebooks.delete("nb_nonexistent")
except Exception as e:
    print(f"Expected error: {e}")

# Example 3: Using async client
from arato_client import AsyncAratoClient


async def delete_notebook_async():
    """Example of deleting a notebook asynchronously."""
    async with AsyncAratoClient(api_key="your-api-key-here") as client:
        # Create a test notebook
        notebook = await client.notebooks.create(
            name="Async Test Notebook to Delete"
        )
        notebook_id = notebook["id"]
        print(f"Created notebook: {notebook_id}")
        
        # Delete it
        await client.notebooks.delete(notebook_id)
        print(f"Deleted notebook {notebook_id} asynchronously!")


# Run the async example
# asyncio.run(delete_notebook_async())


# Important Notes:
# ================
# 1. Deleting a notebook is a permanent action
# 2. It will cascade delete:
#    - All experiments within the notebook
#    - All datasets scoped to the notebook
#    - All runs and evaluations associated with the experiments
# 3. You need Owner or Editor permissions to delete a notebook
# 4. The delete method returns None on success
# 5. On error (404, 403, etc.), it raises an AratoAPIError

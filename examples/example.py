"""Example usage of the AratoClient and AsyncAratoClient to manage notebooks and experiments."""
import asyncio
import os
from arato_client import AratoClient, AsyncAratoClient, NotFoundError

def run_sync_example():
    """Demonstrates the usage of the synchronous AratoClient."""
    print("--- Running Synchronous Example ---")
    try:
        client = AratoClient()

        # 1. Create a new notebook
        print("Creating a new notebook...")
        notebook = client.notebooks.create(
            name="My First Client Notebook",
            description="A notebook created via the Python client.",
            tags=["client-example", "sync"],
        )
        notebook_id = notebook["id"]
        print(f"✅ Notebook '{notebook['name']}' created with ID: {notebook_id}")

        # 2. List all notebooks
        print("\nListing all notebooks...")
        all_notebooks = client.notebooks.list()
        print(f"Found {len(all_notebooks['notebooks'])} notebooks.")

        # 3. Create an experiment within the notebook
        print(f"\nCreating an experiment in notebook {notebook_id}...")
        experiment = client.notebooks.experiments.create(
            notebook_id=notebook_id,
            name="Test Persona Experiment",
            prompt_config={
                "model_id": "gpt-4o-mini",
                "vendor_id": "openai",
                "prompt_template": "Create a persona for {{customer_name}}.",
            },
        )
        experiment_id = experiment["id"]
        print(f"✅ Experiment '{experiment['name']}' created with ID: {experiment_id}")

        # 4. Attempt to fetch a non-existent run to demonstrate error handling
        print("\nAttempting to fetch a non-existent run...")
        try:
            client.notebooks.experiments.runs.retrieve(
                notebook_id=notebook_id,
                experiment_id=experiment_id,
                run_id="run_nonexistent"
            )
        except NotFoundError as e:
            print(f"✅ Successfully caught expected error: {e.message}")

    except (NotFoundError, ValueError, TypeError, KeyError) as e:
        print(f"An unexpected error occurred: {e}")


async def run_async_example():
    """Demonstrates the usage of the asynchronous AsyncAratoClient."""
    print("\n--- Running Asynchronous Example ---")
    try:
        async with AsyncAratoClient() as client:
            print("Listing all notebooks concurrently with dataset listing...")

            # Run tasks concurrently
            notebooks_task = client.notebooks.list()
            datasets_task = client.datasets.list()

            all_notebooks, all_datasets = await asyncio.gather(notebooks_task, datasets_task)

            print(f"✅ Found {len(all_notebooks['notebooks'])} notebooks.")
            print(f"✅ Found {len(all_datasets['datasets'])} global datasets.")

    except (NotFoundError, ValueError, TypeError, KeyError) as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if not os.environ.get("ARATO_API_KEY"):
        print("Error: ARATO_API_KEY environment variable is not set.")
        print("Please set it to your Arato API key to run this example.")
    else:
        run_sync_example()
        asyncio.run(run_async_example())

"""
Async Usage Example - Arato Python SDK

This example demonstrates:
1. Using the AsyncAratoClient for asynchronous operations
2. Concurrent API calls with asyncio.gather()
3. Async context managers
4. Best practices for async workflows
"""

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from arato_client import AsyncAratoClient

# Load environment variables
load_dotenv()


async def main():
    """Main async function demonstrating async SDK usage."""
    
    # Initialize async client with context manager
    async with AsyncAratoClient() as client:
        
        # ====================================================================
        # Part 1: Concurrent Notebook Creation
        # ====================================================================
        
        # Create multiple notebooks concurrently
        notebook_tasks = [
            client.notebooks.create(
                name=f"Async Demo Notebook {i} - {datetime.now().strftime('%H:%M:%S')}",
                description=f"Notebook {i} created asynchronously",
                tags=["async", "demo"]
            )
            for i in range(1, 4)
        ]
        
        start_time = asyncio.get_event_loop().time()
        notebooks = await asyncio.gather(*notebook_tasks)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Created {len(notebooks)} notebooks in {end_time - start_time:.2f}s")
        
        notebook_id = notebooks[0]['id']
        
        # ====================================================================
        # Part 2: Concurrent Dataset Creation
        # ====================================================================
        
        dataset_configs = [
            {
                "name": "Product Reviews",
                "description": "Customer product reviews",
                "tags": ["reviews"],
                "content": [
                    {"product": "Widget A", "rating": 5, "review": "Excellent!"},
                    {"product": "Widget B", "rating": 4, "review": "Very good"},
                ]
            },
            {
                "name": "Support Tickets",
                "description": "Customer support tickets",
                "tags": ["support"],
                "content": [
                    {"ticket_id": "T001", "status": "open", "priority": "high"},
                    {"ticket_id": "T002", "status": "closed", "priority": "medium"},
                ]
            },
            {
                "name": "Sales Data",
                "description": "Monthly sales figures",
                "tags": ["sales"],
                "content": [
                    {"month": "Jan", "revenue": 10000, "units": 150},
                    {"month": "Feb", "revenue": 12000, "units": 180},
                ]
            },
        ]
        
        dataset_tasks = [
            client.notebooks.datasets.create(notebook_id=notebook_id, **config)
            for config in dataset_configs
        ]
        
        start_time = asyncio.get_event_loop().time()
        datasets = await asyncio.gather(*dataset_tasks)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Created {len(datasets)} datasets in {end_time - start_time:.2f}s")
        
        # ====================================================================
        # Part 3: Concurrent Retrieval Operations
        # ====================================================================
        
        # Fetch multiple resources concurrently
        fetch_tasks = [
            client.notebooks.list(),
            client.notebooks.datasets.list(notebook_id=notebook_id),
        ]
        
        start_time = asyncio.get_event_loop().time()
        all_notebooks, all_datasets = await asyncio.gather(*fetch_tasks)
        end_time = asyncio.get_event_loop().time()
        
        print(f"Retrieved data in {end_time - start_time:.2f}s")
        print(f"  Total notebooks: {len(all_notebooks.get('data', []))}")
        print(f"  Datasets in notebook: {len(all_datasets.get('data', []))}")
        
        # ====================================================================
        # Part 4: Async Experiment Creation and Execution
        # ====================================================================
        
        dataset_id = datasets[0]['id']
        
        # Create experiment
        experiment = await client.notebooks.experiments.create(
            notebook_id=notebook_id,
            name="Async Review Analyzer",
            description="Analyzes product reviews asynchronously",
            prompt_config={
                "model_id": "gpt-4o-mini",
                "vendor_id": "openai",
                "prompt_template": (
                    "Analyze this product review and provide a one-sentence summary.\n\n"
                    "Product: {{product}}\n"
                    "Rating: {{rating}}/5\n"
                    "Review: {{review}}\n\n"
                    "Summary:"
                ),
                "model_parameters": {"temperature": 0.7, "max_tokens": 100}
            },
            dataset_id=dataset_id
        )
        print(f"Created experiment: {experiment['id']}")
        
        # Execute run if API key is available
        openai_key = os.environ.get("OPENAI_API_KEY")
        if openai_key:
            run = await client.notebooks.experiments.runs.create(
                notebook_id=notebook_id,
                experiment_id=experiment['id'],
                api_keys={"openai_api_key": openai_key}
            )
            print(f"Run started: {run['id']}")
            
            # Poll for completion
            while True:
                run_details = await client.notebooks.experiments.runs.retrieve(
                    notebook_id=notebook_id,
                    experiment_id=experiment['id'],
                    run_id=run['id']
                )
                if run_details['status'] in ['done', 'failed']:
                    break
                await asyncio.sleep(3)
            
            print(f"Run completed: {run_details['status']}")
        
        # ====================================================================
        # Part 5: Concurrent Multiple Experiment Runs
        # ====================================================================
        
        if openai_key:
            # Create multiple experiments with different temperatures
            experiment_tasks = [
                client.notebooks.experiments.create(
                    notebook_id=notebook_id,
                    name=f"Analyzer (temp={temp})",
                    prompt_config={
                        "model_id": "gpt-4o-mini",
                        "vendor_id": "openai",
                        "prompt_template": "Summarize: {{review}}",
                        "model_parameters": {"temperature": temp, "max_tokens": 50}
                    },
                    dataset_id=dataset_id,
                    color_index=idx
                )
                for idx, temp in enumerate([0.3, 0.7, 1.0])
            ]
            
            experiments = await asyncio.gather(*experiment_tasks)
            print(f"Created {len(experiments)} experiments concurrently")
            
            # Run all experiments concurrently
            run_tasks = [
                client.notebooks.experiments.runs.create(
                    notebook_id=notebook_id,
                    experiment_id=exp['id'],
                    api_keys={"openai_api_key": openai_key}
                )
                for exp in experiments
            ]
            
            runs = await asyncio.gather(*run_tasks)
            print(f"Started {len(runs)} runs concurrently")
            
            # Wait for all runs to complete
            async def wait_for_run(exp_id, run_id):
                while True:
                    details = await client.notebooks.experiments.runs.retrieve(
                        notebook_id=notebook_id,
                        experiment_id=exp_id,
                        run_id=run_id
                    )
                    if details['status'] in ['done', 'failed']:
                        return details
                    await asyncio.sleep(3)
            
            wait_tasks = [
                wait_for_run(exp['id'], run['id'])
                for exp, run in zip(experiments, runs)
            ]
            
            completed_runs = await asyncio.gather(*wait_tasks)
            print(f"All {len(completed_runs)} runs completed")
            
            for i, run_details in enumerate(completed_runs, 1):
                print(f"  Run {i}: {run_details['status']}")
        
        print(f"\nView notebook: https://app.arato.io/notebooks/{notebook_id}")


if __name__ == "__main__":
    asyncio.run(main())

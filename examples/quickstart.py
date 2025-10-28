"""
Quickstart Example - Arato Python SDK

This example demonstrates the basic workflow:
1. Initialize the Arato client
2. Create a notebook
3. Create a dataset
4. Create an experiment
5. Execute a run
"""

import os
import time
from dotenv import load_dotenv
from arato_client import AratoClient

# Load environment variables
load_dotenv()

# Initialize the Arato client
client = AratoClient()

# Step 1: Create a notebook
notebook = client.notebooks.create(
    name="Quickstart Example",
    description="A simple example to get started with Arato SDK",
    tags=["quickstart", "example"]
)
notebook_id = notebook['id']
print(f"Created notebook: {notebook_id}")

# Step 2: Create a dataset
dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Sample Questions",
    description="A small dataset with example questions",
    content=[
        {"question": "What is the capital of France?", "category": "geography"},
        {"question": "Who wrote Romeo and Juliet?", "category": "literature"},
        {"question": "What is 2 + 2?", "category": "math"},
        {"question": "What is the speed of light?", "category": "science"},
    ]
)
print(f"Created dataset with {len(dataset['content'])} rows")

# Step 3: Create an experiment
experiment = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Question Answerer",
    description="Answers questions using AI",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a knowledgeable assistant. Answer the following question "
            "concisely and accurately.\n\n"
            "Question: {{question}}"
        ),
        "model_parameters": {
            "temperature": 0.7,
            "max_tokens": 100
        }
    },
    dataset_id=dataset['id']
)
print(f"Created experiment: {experiment['id']}")

# Step 4: Execute a run (if API key is available)
openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    print("Set OPENAI_API_KEY to execute runs")
else:
    run = client.notebooks.experiments.runs.create(
        notebook_id=notebook_id,
        experiment_id=experiment['id'],
        api_keys={"openai_api_key": openai_key}
    )
    print(f"Run started: {run['id']}")
    
    # Wait for completion
    while True:
        run_details = client.notebooks.experiments.runs.retrieve(
            notebook_id=notebook_id,
            experiment_id=experiment['id'],
            run_id=run['id']
        )
        if run_details['status'] in ['done', 'failed']:
            break
        time.sleep(3)
    
    print(f"Run completed with status: {run_details['status']}")
    
    # Display sample results
    if run_details['status'] == 'done':
        for row in run_details['content'][:2]:  # Show first 2
            print(f"Q: {row['question']}")
            print(f"A: {row['response']}\n")

print(f"View notebook: https://app.arato.io/notebooks/{notebook_id}")

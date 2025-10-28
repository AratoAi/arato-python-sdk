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

import os
import time
from dotenv import load_dotenv
from arato_client import AratoClient

# Load environment variables
load_dotenv()

# Initialize the Arato client
# API key is automatically loaded from ARATO_API_KEY environment variable
client = AratoClient()

print("🚀 Arato SDK Quickstart Example")
print("=" * 60)

# Step 1: Create a notebook
print("\n📓 Step 1: Creating a notebook...")
notebook = client.notebooks.create(
    name="Quickstart Example",
    description="A simple example to get started with Arato SDK",
    tags=["quickstart", "example"]
)
notebook_id = notebook['id']
print(f"✅ Created notebook: {notebook_id}")

# Step 2: Create a dataset
print("\n📊 Step 2: Creating a dataset...")
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
dataset_id = dataset['id']
print(f"✅ Created dataset with {len(dataset['content'])} rows")

# Step 3: Create an experiment
print("\n🧪 Step 3: Creating an experiment...")
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
    dataset_id=dataset_id
)
experiment_id = experiment['id']
print(f"✅ Created experiment: {experiment_id}")

# Step 4: Execute a run
print("\n▶️  Step 4: Executing a run...")
openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
    print("   Set your OpenAI API key to execute runs")
else:
    run = client.notebooks.experiments.runs.create(
        notebook_id=notebook_id,
        experiment_id=experiment_id,
        api_keys={"openai_api_key": openai_key}
    )
    run_id = run['id']
    print(f"✅ Run started: {run_id}")
    
    # Step 5: Wait for completion and get results
    print("\n⏳ Step 5: Waiting for run to complete...")
    while True:
        run_details = client.notebooks.experiments.runs.retrieve(
            notebook_id=notebook_id,
            experiment_id=experiment_id,
            run_id=run_id
        )
        
        status = run_details['status']
        if status in ['done', 'failed']:
            break
        
        print(f"   Status: {status}")
        time.sleep(3)
    
    # Display results
    if status == 'done':
        print("\n✅ Run completed successfully!")
        print("\n📋 Results:")
        print("=" * 60)
        
        for i, row in enumerate(run_details['content'], 1):
            print(f"\n{i}. Question: {row['question']}")
            print(f"   Category: {row['category']}")
            print(f"   Answer: {row['response']}")
            print(f"   Tokens: {row.get('tokens_in', 0)} in, {row.get('tokens_out', 0)} out")
    else:
        print(f"\n❌ Run failed with status: {status}")

print("\n" + "=" * 60)
print("🎉 Quickstart example completed!")
print(f"\nView your notebook at: https://app.arato.io/notebooks/{notebook_id}")

# Arato Python SDK Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Authentication](#authentication)
5. [Client Reference](#client-reference)
6. [Resource Objects](#resource-objects)
   - [Notebooks](#notebooks)
   - [Datasets](#datasets)
   - [Experiments](#experiments)
   - [Runs](#runs)
   - [Evaluations (Evals)](#evaluations-evals)
7. [Tutorial: Building an LLM-as-a-Judge System](#tutorial-building-an-llm-as-a-judge-system)
8. [Error Handling](#error-handling)
9. [Best Practices](#best-practices)
10. [API Reference](#api-reference)

---

## Overview

The Arato Python SDK provides a comprehensive interface for building, managing, and evaluating AI/ML experiments. It enables you to:

- **Organize Work**: Create notebooks to group related experiments and datasets
- **Manage Data**: Store and version datasets for experimentation
- **Run Experiments**: Execute AI models with configurable prompts and parameters
- **Evaluate Results**: Automatically assess experiment outputs using AI judges
- **Track Progress**: Monitor runs and analyze performance metrics

### Key Features

- **Synchronous and Asynchronous Support**: Choose between `AratoClient` (sync) or `AsyncAratoClient` (async)
- **Type-Safe**: Full type hints for better IDE support and error detection
- **Resource Hierarchy**: Intuitive organization with notebooks containing experiments and datasets
- **Flexible Evaluation**: Support for Binary, Numeric, and Classification evaluations
- **Error Handling**: Comprehensive exception handling with detailed error messages

---

## Installation

Install the Arato SDK using pip:

```bash
pip install arato-client
```

For development installations:

```bash
git clone https://github.com/AratoAi/arato-python-sdk.git
cd arato-python-sdk
pip install -e .
```

### Requirements

- Python 3.7+
- httpx
- python-dotenv (optional, for environment variable management)

---

## Quick Start

```python
from arato_client import AratoClient
import os

# Initialize the client
client = AratoClient(api_key=os.environ.get("ARATO_API_KEY"))

# Create a notebook
notebook = client.notebooks.create(
    name="My First Notebook",
    description="Getting started with Arato"
)

# Create a dataset
dataset = client.notebooks.datasets.create(
    notebook_id=notebook['id'],
    name="Test Data",
    content=[
        {"input": "Hello", "expected": "Hi there!"},
        {"input": "How are you?", "expected": "I'm doing well!"}
    ]
)

print(f"✅ Created notebook: {notebook['id']}")
print(f"✅ Created dataset: {dataset['id']}")
```

---

## Authentication

The Arato SDK requires an API key for authentication. You can provide it in three ways:

### 1. Environment Variable (Recommended)

```bash
export ARATO_API_KEY="your-api-key-here"
```

```python
from arato_client import AratoClient

# API key is automatically loaded from ARATO_API_KEY
client = AratoClient()
```

### 2. Direct Parameter

```python
from arato_client import AratoClient

client = AratoClient(api_key="your-api-key-here")
```

### 3. Using .env File

Create a `.env` file:
```
ARATO_API_KEY=your-api-key-here
OPENAI_API_KEY=your-openai-key-here
```

Load it in your code:
```python
from dotenv import load_dotenv
from arato_client import AratoClient

load_dotenv()
client = AratoClient()  # Automatically uses ARATO_API_KEY from .env
```

---

## Client Reference

### AratoClient (Synchronous)

The main client for making synchronous API requests.

```python
from arato_client import AratoClient

client = AratoClient(
    api_key="your-api-key",           # Optional if set in environment
    base_url="https://api.arato.ai/api/v1",  # Optional, defaults to production
    timeout=60.0,                      # Request timeout in seconds
    max_retries=2                      # Maximum retry attempts
)
```

#### Parameters

- **api_key** (str, optional): Your Arato API key. Defaults to `ARATO_API_KEY` environment variable.
- **base_url** (str, optional): The API base URL. Defaults to production server.
- **timeout** (float, optional): Request timeout in seconds. Default: 60.0
- **max_retries** (int, optional): Maximum number of retry attempts for transient errors. Default: 2
- **http_client** (httpx.Client, optional): Custom HTTP client for advanced configurations.

#### Context Manager Support

```python
with AratoClient() as client:
    notebooks = client.notebooks.list()
    # Client is automatically closed after the block
```

#### Resource Access

The client provides access to resources through attributes:

- **client.notebooks**: Notebook operations
- **client.datasets**: Global dataset operations
- **client.notebooks.experiments**: Experiment operations (requires notebook_id)
- **client.notebooks.experiments.runs**: Run operations (requires experiment_id)
- **client.notebooks.experiments.evals**: Evaluation operations (requires experiment_id)

### AsyncAratoClient (Asynchronous)

For asynchronous operations using `asyncio`:

```python
import asyncio
from arato_client import AsyncAratoClient

async def main():
    async with AsyncAratoClient() as client:
        notebooks = await client.notebooks.list()
        print(f"Found {len(notebooks['data'])} notebooks")

asyncio.run(main())
```

All methods on `AsyncAratoClient` are async and must be awaited.

---

## Resource Objects

### Notebooks

Notebooks are top-level containers for organizing experiments, datasets, and evaluations.

#### List Notebooks

Retrieve all notebooks accessible to your account.

```python
notebooks = client.notebooks.list()

for notebook in notebooks['data']:
    print(f"Notebook: {notebook['name']} (ID: {notebook['id']})")
```

**Response Structure:**
```python
{
    "data": [
        {
            "id": "nb_abc123",
            "name": "My Notebook",
            "description": "A description",
            "tags": ["ml", "experiment"],
            "created_at": "2025-10-27T10:00:00Z",
            "updated_at": "2025-10-27T10:00:00Z",
            "_links": {
                "experiments": {"href": "/notebooks/nb_abc123/experiments"},
                "datasets": {"href": "/notebooks/nb_abc123/datasets"}
            }
        }
    ]
}
```

#### Create Notebook

Create a new notebook.

```python
notebook = client.notebooks.create(
    name="Customer Support Analysis",
    description="Analyzing customer support interactions",
    tags=["support", "nlp", "production"]
)

notebook_id = notebook['id']
```

**Parameters:**
- **name** (str, required): Name of the notebook
- **description** (str, optional): Detailed description
- **tags** (List[str], optional): Tags for categorization

**Returns:** Dictionary containing the created notebook object.

#### Retrieve Notebook

Get a specific notebook by ID.

```python
notebook = client.notebooks.retrieve(notebook_id="nb_abc123")

print(f"Notebook: {notebook['name']}")
print(f"Created: {notebook['created_at']}")
```

**Parameters:**
- **notebook_id** (str, required): The unique identifier for the notebook

**Returns:** Dictionary containing the notebook object.

---

### Datasets

Datasets are collections of data used for experiments. They can be global (accessible across notebooks) or notebook-scoped.

#### Global Datasets

##### List Global Datasets

```python
datasets = client.datasets.list()

for dataset in datasets['data']:
    print(f"Dataset: {dataset['name']}")
```

##### Create Global Dataset

```python
dataset = client.datasets.create(
    name="Common Queries Dataset",
    description="Frequently asked questions",
    tags=["faq", "global"],
    content=[
        {"query": "What are your hours?", "category": "business_info"},
        {"query": "How do I reset my password?", "category": "technical"},
        {"query": "What is your refund policy?", "category": "policy"}
    ]
)
```

**Parameters:**
- **name** (str, required): Name of the dataset
- **description** (str, optional): Description of the dataset
- **tags** (List[str], optional): Tags for organization
- **content** (List[Dict], optional): Array of data objects

##### Retrieve Global Dataset

```python
dataset = client.datasets.retrieve(dataset_id="ds_xyz789")
```

#### Notebook-Scoped Datasets

##### List Notebook Datasets

```python
datasets = client.notebooks.datasets.list(notebook_id="nb_abc123")
```

##### Create Notebook Dataset

```python
dataset = client.notebooks.datasets.create(
    notebook_id="nb_abc123",
    name="Experiment Test Cases",
    description="Test cases for toxicity detection",
    content=[
        {"query": "I love this!", "ground_truth": "positive"},
        {"query": "This is terrible", "ground_truth": "negative"}
    ]
)

dataset_id = dataset['id']
```

**Parameters:**
- **notebook_id** (str, required): The notebook ID
- **name** (str, required): Name of the dataset
- **description** (str, optional): Description
- **tags** (List[str], optional): Tags
- **content** (List[Dict], optional): Array of data rows

##### Retrieve Notebook Dataset

```python
dataset = client.notebooks.datasets.retrieve(
    notebook_id="nb_abc123",
    dataset_id="ds_xyz789"
)

print(f"Dataset has {len(dataset['content'])} rows")
```

#### Dataset Content Structure

Dataset content is a list of dictionaries. Each dictionary represents one row:

```python
content = [
    {
        "input": "What is AI?",
        "expected_output": "Artificial Intelligence is...",
        "category": "definition"
    },
    {
        "input": "How does ML work?",
        "expected_output": "Machine Learning works by...",
        "category": "explanation"
    }
]
```

Column names (keys) can be referenced in experiment prompt templates using `{{column_name}}`.

---

### Experiments

Experiments define AI model configurations, prompts, and execution parameters.

#### List Experiments

```python
experiments = client.notebooks.experiments.list(notebook_id="nb_abc123")

for exp in experiments['data']:
    print(f"Experiment: {exp['name']}")
    print(f"  Model: {exp['prompt_config']['model_id']}")
```

#### Create Experiment

```python
experiment = client.notebooks.experiments.create(
    notebook_id="nb_abc123",
    name="Sentiment Analyzer",
    description="Analyzes sentiment of user feedback",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a sentiment analyzer. Analyze the following text and respond "
            "with only one word: positive, negative, or neutral.\n\n"
            "Text: {{input}}"
        ),
        "model_parameters": {
            "temperature": 0.1,
            "max_tokens": 10
        }
    },
    dataset_id="ds_xyz789",
    color_index=2
)

experiment_id = experiment['id']
```

**Parameters:**
- **notebook_id** (str, required): The notebook ID
- **name** (str, required): Name of the experiment
- **prompt_config** (Dict, required): Prompt configuration (see below)
- **description** (str, optional): Description
- **prompt_type** (str, optional): Type of prompt. Default: "generating_prompt"
- **color_index** (int, optional): Color for UI display (0-7)
- **dataset_id** (str, optional): Associated dataset ID

#### Prompt Configuration

The `prompt_config` object defines how the AI model is invoked:

```python
prompt_config = {
    "model_id": "gpt-4o-mini",           # Model identifier
    "vendor_id": "openai",               # AI provider (openai, anthropic, etc.)
    "prompt_template": "Your prompt...", # Prompt with variables
    "model_parameters": {                # Model-specific parameters
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1.0
    }
}
```

**Prompt Template Variables:**

You can reference dataset columns using double curly braces:

```python
prompt_template = "Analyze this query: {{query}}\n\nExpected answer: {{expected_answer}}"
```

When a run executes, `{{query}}` and `{{expected_answer}}` are replaced with values from each row in the dataset.

**Message Array Format:**

For more complex prompts, use message arrays:

```python
prompt_template = [
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
        "content": "{{user_query}}"
    }
]
```

#### Retrieve Experiment

```python
experiment = client.notebooks.experiments.retrieve(
    notebook_id="nb_abc123",
    experiment_id="exp_def456"
)
```

#### Update Experiment

Update an existing experiment:

```python
updated_experiment = client.notebooks.experiments.update(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    name="Improved Sentiment Analyzer",
    prompt_config={
        "model_id": "gpt-4o",
        "vendor_id": "openai",
        "prompt_template": "Enhanced prompt...",
        "model_parameters": {"temperature": 0.5}
    },
    dataset_id="ds_new123"
)
```

**Parameters:**
- **notebook_id** (str, required): The notebook ID
- **experiment_id** (str, required): The experiment ID
- **name** (str, optional): New name
- **description** (str, optional): New description
- **prompt_config** (Dict, optional): Updated prompt configuration
- **color_index** (int, optional): New color index
- **dataset_id** (str, optional): New dataset ID

---

### Runs

Runs execute experiments against datasets and collect results.

#### List Runs

```python
runs = client.notebooks.experiments.runs.list(
    notebook_id="nb_abc123",
    experiment_id="exp_def456"
)

for run in runs['data']:
    print(f"Run #{run['run_number']}: {run['status']}")
```

#### Create and Execute Run

```python
import os

run = client.notebooks.experiments.runs.create(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    api_keys={
        "openai_api_key": os.environ.get("OPENAI_API_KEY")
    },
    callback_url="https://myapp.com/webhook/run-complete"  # Optional
)

run_id = run['id']
print(f"Run created: {run_id}")
print(f"Status: {run['status']}")
```

**Parameters:**
- **notebook_id** (str, required): The notebook ID
- **experiment_id** (str, required): The experiment ID
- **api_keys** (Dict[str, str], required): API keys for AI providers
- **callback_url** (str, optional): Webhook URL for status updates

**API Keys Format:**
```python
api_keys = {
    "openai_api_key": "sk-...",
    "anthropic_api_key": "sk-ant-...",
    # Add other provider keys as needed
}
```

#### Retrieve Run

Get the status and results of a specific run:

```python
run_details = client.notebooks.experiments.runs.retrieve(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    run_id="run_ghi789"
)

print(f"Status: {run_details['status']}")
print(f"Run Number: {run_details['run_number']}")

# Access results
for row in run_details.get('content', []):
    print(f"Input: {row.get('input')}")
    print(f"Response: {row.get('response')}")
    print(f"Tokens: {row.get('tokens_in')} in, {row.get('tokens_out')} out")
```

**Run Status Values:**
- `pending`: Run is queued
- `running`: Run is in progress
- `done`: Run completed successfully
- `failed`: Run encountered an error

**Run Result Structure:**
```python
{
    "id": "run_ghi789",
    "run_number": 1,
    "status": "done",
    "created_at": "2025-10-27T10:00:00Z",
    "content": [
        {
            # Original dataset columns
            "input": "What is AI?",
            "expected": "...",
            
            # Generated response
            "response": "Artificial Intelligence is...",
            
            # Token usage
            "tokens_in": 50,
            "tokens_out": 75,
            "finish_reason": "stop",
            
            # Evaluation results (if evals are configured)
            "evals": [
                {
                    "eval_id": "eval_abc",
                    "type": "Binary",
                    "result": 1  # 1 = pass, 0 = fail
                }
            ]
        }
    ]
}
```

#### Polling for Run Completion

```python
import time

# Create run
run = client.notebooks.experiments.runs.create(
    notebook_id=notebook_id,
    experiment_id=experiment_id,
    api_keys={"openai_api_key": openai_key}
)

run_id = run['id']

# Poll for completion
while True:
    run_details = client.notebooks.experiments.runs.retrieve(
        notebook_id=notebook_id,
        experiment_id=experiment_id,
        run_id=run_id
    )
    
    status = run_details['status']
    print(f"Status: {status}")
    
    if status in ['done', 'failed']:
        break
    
    time.sleep(5)  # Wait 5 seconds before checking again

# Process results
if status == 'done':
    for row in run_details['content']:
        print(f"Response: {row['response']}")
```

---

### Evaluations (Evals)

Evaluations automatically assess experiment outputs using AI judges or rule-based logic.

#### Types of Evaluations

1. **Binary**: Yes/No judgments (e.g., "Is this toxic?")
2. **Numeric**: Scored evaluations with ranges (e.g., 1-10 quality score)
3. **Classification**: Multi-class categorization (e.g., Safe/Moderate/High)

#### List Evaluations

```python
evals = client.notebooks.experiments.evals.list(
    notebook_id="nb_abc123",
    experiment_id="exp_def456"
)

for eval_obj in evals['data']:
    print(f"Eval: {eval_obj['name']} ({eval_obj['eval_type']})")
```

#### Create Binary Evaluation

Binary evaluations produce yes/no judgments.

```python
binary_eval = client.notebooks.experiments.evals.create(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    name="Toxicity Detector",
    eval_type="Binary",
    context="response",  # What to evaluate: "response", "query", "prompt_and_response"
    fail_on_positive=True,  # True = "yes" means failure
    prompt=(
        "Is the following text toxic? Respond with only 'yes' or 'no'.\n\n"
        "Text: {{response}}"
    )
)
```

**Parameters:**
- **notebook_id** (str, required): The notebook ID
- **experiment_id** (str, required): The experiment ID
- **name** (str, required): Name of the evaluation
- **eval_type** (str, required): "Binary"
- **context** (str, optional): What to evaluate. Default: "prompt_and_response"
  - `"response"`: Only the model's response
  - `"query"`: Only the input query/prompt
  - `"prompt_and_response"`: Both input and output
- **prompt** (str, optional): Prompt for the judge model
- **fail_on_positive** (bool, optional): If True, "yes" = fail, "no" = pass

**Result Format:**
```python
{
    "eval_id": "eval_abc",
    "type": "Binary",
    "result": 1,  # 1 = pass, 0 = fail
    "raw_output": "no"
}
```

#### Create Numeric Evaluation

Numeric evaluations score outputs on a scale with defined ranges.

```python
numeric_eval = client.notebooks.experiments.evals.create(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    name="Quality Score",
    eval_type="Numeric",
    context="response",
    prompt=(
        "Rate the quality of this response on a scale of 1-10.\n\n"
        "Response: {{response}}"
    ),
    ranges=[
        {
            "range_id": "poor",
            "min_score": 1.0,
            "max_score": 4.0,
            "is_pass": False
        },
        {
            "range_id": "acceptable",
            "min_score": 4.1,
            "max_score": 7.0,
            "is_pass": True
        },
        {
            "range_id": "excellent",
            "min_score": 7.1,
            "max_score": 10.0,
            "is_pass": True
        }
    ]
)
```

**Parameters:**
- **ranges** (List[Dict], required): Score range definitions
  - **range_id** (str): Identifier for the range
  - **min_score** (float): Minimum score (inclusive)
  - **max_score** (float): Maximum score (inclusive)
  - **is_pass** (bool): Whether this range indicates success

**Result Format:**
```python
{
    "eval_id": "eval_abc",
    "type": "Numeric",
    "result": 1,  # 1 if is_pass=True for the range, 0 otherwise
    "score": 8.5,
    "range_id": "excellent",
    "raw_output": "8.5"
}
```

#### Create Classification Evaluation

Classification evaluations categorize outputs into predefined classes.

```python
classification_eval = client.notebooks.experiments.evals.create(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    name="Sentiment Classifier",
    eval_type="Classification",
    context="response",
    prompt=(
        "Classify the sentiment of this text. Respond with only one word: "
        "Positive, Neutral, or Negative.\n\n"
        "Text: {{response}}"
    ),
    classes=[
        {
            "title": "Positive",
            "is_pass": True,
            "color": "green"
        },
        {
            "title": "Neutral",
            "is_pass": True,
            "color": "yellow"
        },
        {
            "title": "Negative",
            "is_pass": False,
            "color": "red"
        }
    ]
)
```

**Parameters:**
- **classes** (List[Dict], required): Classification class definitions
  - **title** (str): Name of the class
  - **is_pass** (bool): Whether this class indicates success
  - **color** (str, optional): Color for UI display

**Result Format:**
```python
{
    "eval_id": "eval_abc",
    "type": "Classification",
    "result": 1,  # 1 if is_pass=True for the class, 0 otherwise
    "title": "Positive",
    "raw_output": "Positive"
}
```

#### Retrieve Evaluation

```python
eval_obj = client.notebooks.experiments.evals.retrieve(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    eval_id="eval_jkl012"
)

print(f"Eval Type: {eval_obj['eval_type']}")
print(f"Context: {eval_obj['context']}")
```

#### Update Evaluation

```python
updated_eval = client.notebooks.experiments.evals.update(
    notebook_id="nb_abc123",
    experiment_id="exp_def456",
    eval_id="eval_jkl012",
    name="Updated Toxicity Detector",
    context="prompt_and_response",
    fail_on_positive=False
)
```

---

## Tutorial: Building an LLM-as-a-Judge System

This tutorial demonstrates how to build a complete LLM-as-a-Judge system for content moderation. We'll create a toxicity detection system that evaluates user queries at scale.

### Overview

We'll build a system that:
1. Accepts user queries as input
2. Generates responses using an AI assistant
3. Evaluates both queries and responses for toxicity
4. Uses both binary and multi-class classification judges
5. Flags problematic content for review

### Step 1: Initialize and Setup

First, set up the environment and create a notebook to organize our work.

```python
import os
from datetime import datetime
from dotenv import load_dotenv
from arato_client import AratoClient

# Load environment variables
load_dotenv()

# Initialize client
client = AratoClient()

# Create a notebook for our moderation system
notebook = client.notebooks.create(
    name=f"Content Moderation System - {datetime.now().strftime('%Y-%m-%d')}",
    description="AI-powered content moderation with LLM judges",
    tags=["moderation", "safety", "llm-judge"]
)

notebook_id = notebook['id']
print(f"✅ Created notebook: {notebook_id}")
```

### Step 2: Create Test Dataset

Create a dataset with example user queries for testing our judge.

```python
# Create a test dataset with labeled examples
test_dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Judge Test Cases",
    description="Labeled examples for validating the toxicity judge",
    content=[
        {"query": "Thank you so much for your help!", "expected_toxicity": "Safe"},
        {"query": "You're an idiot if you believe that", "expected_toxicity": "Moderate"},
        {"query": "I'm going to hurt you", "expected_toxicity": "High"},
        {"query": "This feature is amazing", "expected_toxicity": "Safe"},
        {"query": "Your service is garbage", "expected_toxicity": "Moderate"},
        {"query": "I will destroy you and your family", "expected_toxicity": "High"},
    ]
)

test_dataset_id = test_dataset['id']
print(f"✅ Created test dataset with {len(test_dataset['content'])} examples")
```

### Step 3: Create Binary Judge Experiment

Create an experiment that acts as a binary toxicity judge.

```python
# Create binary judge experiment
binary_judge_experiment = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Binary Toxicity Judge",
    description="Simple yes/no toxicity detection",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a toxicity judge. Determine if the following query is toxic. "
            "A query is toxic if it contains hate speech, threats, or harassment. "
            "Respond with only 'yes' or 'no'.\n\n"
            "Query: {{query}}"
        ),
        "model_parameters": {
            "temperature": 0.1,
            "max_tokens": 10
        }
    },
    dataset_id=test_dataset_id,
    color_index=0
)

binary_judge_id = binary_judge_experiment['id']
print(f"✅ Created binary judge experiment: {binary_judge_id}")
```

### Step 4: Run and Validate Binary Judge

Execute the binary judge on our test data.

```python
import time

# Run the binary judge
run = client.notebooks.experiments.runs.create(
    notebook_id=notebook_id,
    experiment_id=binary_judge_id,
    api_keys={"openai_api_key": os.environ.get("OPENAI_API_KEY")}
)

run_id = run['id']
print(f"⏳ Running binary judge (Run ID: {run_id})...")

# Poll for completion
while True:
    run_details = client.notebooks.experiments.runs.retrieve(
        notebook_id=notebook_id,
        experiment_id=binary_judge_id,
        run_id=run_id
    )
    
    if run_details['status'] in ['done', 'failed']:
        break
    
    time.sleep(3)

# Analyze results
print("\n📊 Binary Judge Results:")
print("=" * 60)

correct = 0
total = 0

for row in run_details['content']:
    query = row['query']
    expected = row['expected_toxicity']
    response = row['response'].strip().lower()
    
    # Convert to binary
    is_toxic = response == 'yes'
    expected_toxic = expected in ['Moderate', 'High']
    
    is_correct = is_toxic == expected_toxic
    if is_correct:
        correct += 1
    total += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{status} '{query[:50]}...' -> {response} (expected: {expected})")

accuracy = (correct / total * 100) if total > 0 else 0
print(f"\n📈 Accuracy: {correct}/{total} = {accuracy:.1f}%")
```

### Step 5: Create Multi-Level Classification Judge

Binary judgments are limited. Let's create a more nuanced judge with three toxicity levels.

```python
# Create classification judge experiment
classification_judge_experiment = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Multi-Level Toxicity Classifier",
    description="Three-level toxicity classification: Safe, Moderate, High",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are an advanced toxicity classifier. Classify the toxicity level "
            "of the following query.\n\n"
            "Guidelines:\n"
            "- Safe: Polite, constructive, or neutral content\n"
            "- Moderate: Rude or mildly offensive content\n"
            "- High: Hate speech, threats, or severe harassment\n\n"
            "Respond with only one word: Safe, Moderate, or High.\n\n"
            "Query: {{query}}"
        ),
        "model_parameters": {
            "temperature": 0.1,
            "max_tokens": 10
        }
    },
    dataset_id=test_dataset_id,
    color_index=1
)

classification_judge_id = classification_judge_experiment['id']
print(f"✅ Created classification judge: {classification_judge_id}")
```

### Step 6: Run and Validate Classification Judge

```python
# Run the classification judge
run = client.notebooks.experiments.runs.create(
    notebook_id=notebook_id,
    experiment_id=classification_judge_id,
    api_keys={"openai_api_key": os.environ.get("OPENAI_API_KEY")}
)

run_id = run['id']
print(f"⏳ Running classification judge...")

# Poll for completion
while True:
    run_details = client.notebooks.experiments.runs.retrieve(
        notebook_id=notebook_id,
        experiment_id=classification_judge_id,
        run_id=run_id
    )
    
    if run_details['status'] in ['done', 'failed']:
        break
    
    time.sleep(3)

# Analyze results with per-class breakdown
print("\n📊 Classification Judge Results:")
print("=" * 60)

from collections import defaultdict

correct = 0
total = 0
confusion = defaultdict(lambda: defaultdict(int))

for row in run_details['content']:
    query = row['query']
    expected = row['expected_toxicity']
    predicted = row['response'].strip()
    
    is_correct = predicted == expected
    if is_correct:
        correct += 1
    total += 1
    
    confusion[expected][predicted] += 1
    
    status = "✅" if is_correct else "❌"
    print(f"{status} '{query[:40]}...'")
    print(f"   Expected: {expected}, Predicted: {predicted}")

accuracy = (correct / total * 100) if total > 0 else 0
print(f"\n📈 Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")

# Show per-class accuracy
print("\n📊 Per-Class Performance:")
for actual_class in ['Safe', 'Moderate', 'High']:
    class_total = sum(confusion[actual_class].values())
    class_correct = confusion[actual_class][actual_class]
    class_acc = (class_correct / class_total * 100) if class_total > 0 else 0
    print(f"   {actual_class}: {class_correct}/{class_total} = {class_acc:.1f}%")
```

### Step 7: Create Production System

Now let's build the production system that generates responses and evaluates them.

```python
# Create production dataset (no labels)
production_dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Production User Queries",
    description="Real user queries to process and evaluate",
    content=[
        {"query": "How do I reset my password?"},
        {"query": "Your app is complete trash and waste of time"},
        {"query": "Can you help me understand this feature?"},
        {"query": "I hate you all, you should be fired"},
        {"query": "The new update is fantastic!"},
        {"query": "You people are incompetent fools"},
        {"query": "What are your business hours?"},
        {"query": "I'm going to report you to the authorities for this"},
        {"query": "Thanks for the quick support!"},
        {"query": "This is the worst service I've ever experienced"},
    ]
)

production_dataset_id = production_dataset['id']
print(f"✅ Created production dataset: {production_dataset_id}")
```

### Step 8: Create Assistant Experiment

Create an experiment that generates responses to user queries.

```python
# Create assistant experiment
assistant_experiment = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Customer Support Assistant",
    description="AI assistant that responds to user queries",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a helpful customer support assistant. "
            "Provide a brief, professional response to the following query.\n\n"
            "User Query: {{query}}"
        ),
        "model_parameters": {
            "temperature": 0.7,
            "max_tokens": 150
        }
    },
    dataset_id=production_dataset_id,
    color_index=2
)

assistant_experiment_id = assistant_experiment['id']
print(f"✅ Created assistant experiment: {assistant_experiment_id}")
```

### Step 9: Add Evaluations to Production System

Configure automatic evaluation of queries and responses.

```python
# Add binary toxicity evaluation for input queries
binary_eval = client.notebooks.experiments.evals.create(
    notebook_id=notebook_id,
    experiment_id=assistant_experiment_id,
    name="Input Toxicity Check",
    eval_type="Binary",
    context="query",  # Evaluate the input query
    fail_on_positive=True,
    prompt=(
        "Is the following user query toxic (containing hate speech, threats, "
        "or harassment)? Respond with only 'yes' or 'no'.\n\n"
        "Query: {{query}}"
    )
)

print(f"✅ Added binary toxicity eval: {binary_eval['id']}")

# Add classification evaluation for more detailed analysis
classification_eval = client.notebooks.experiments.evals.create(
    notebook_id=notebook_id,
    experiment_id=assistant_experiment_id,
    name="Toxicity Level Classification",
    eval_type="Classification",
    context="query",
    prompt=(
        "Classify the toxicity level of this query.\n\n"
        "Guidelines:\n"
        "- Safe: Polite, constructive, or neutral\n"
        "- Moderate: Rude or mildly offensive\n"
        "- High: Hate speech, threats, or severe harassment\n\n"
        "Respond with only: Safe, Moderate, or High\n\n"
        "Query: {{query}}"
    ),
    classes=[
        {"title": "Safe", "is_pass": True, "color": "green"},
        {"title": "Moderate", "is_pass": False, "color": "yellow"},
        {"title": "High", "is_pass": False, "color": "red"}
    ]
)

print(f"✅ Added classification eval: {classification_eval['id']}")
```

### Step 10: Run Production System

Execute the complete system with automatic evaluations.

```python
# Run the production system
run = client.notebooks.experiments.runs.create(
    notebook_id=notebook_id,
    experiment_id=assistant_experiment_id,
    api_keys={"openai_api_key": os.environ.get("OPENAI_API_KEY")}
)

run_id = run['id']
print(f"⏳ Running production system...")

# Poll for completion
while True:
    run_details = client.notebooks.experiments.runs.retrieve(
        notebook_id=notebook_id,
        experiment_id=assistant_experiment_id,
        run_id=run_id
    )
    
    if run_details['status'] in ['done', 'failed']:
        break
    
    time.sleep(3)

# Analyze production results
print("\n" + "=" * 80)
print("📊 PRODUCTION RESULTS WITH AUTOMATIC EVALUATIONS")
print("=" * 80)

binary_stats = {"toxic": 0, "safe": 0}
classification_stats = {"Safe": 0, "Moderate": 0, "High": 0}
flagged_queries = []

for idx, row in enumerate(run_details['content'], 1):
    query = row['query']
    response = row['response']
    
    print(f"\n{'─' * 60}")
    print(f"Query {idx}: \"{query}\"")
    print(f"Response: \"{response[:80]}...\"")
    
    # Process evaluation results
    if row.get('evals'):
        for eval_result in row['evals']:
            eval_type = eval_result.get('type')
            
            if eval_type == 'Binary':
                is_toxic = eval_result.get('result') == 0  # 0 = fail (toxic)
                result_str = "🚨 TOXIC" if is_toxic else "✅ SAFE"
                binary_stats["toxic" if is_toxic else "safe"] += 1
                print(f"  Binary Judge: {result_str}")
                
                if is_toxic:
                    flagged_queries.append((idx, query, "Binary"))
            
            elif eval_type == 'Classification':
                level = eval_result.get('title', 'Unknown')
                is_pass = eval_result.get('result') == 1
                
                classification_stats[level] = classification_stats.get(level, 0) + 1
                
                icons = {"Safe": "🟢", "Moderate": "🟡", "High": "🔴"}
                icon = icons.get(level, "❓")
                print(f"  Classification: {icon} {level}")
                
                if level in ['Moderate', 'High']:
                    flagged_queries.append((idx, query, level))

# Summary statistics
print(f"\n{'=' * 80}")
print("📈 SUMMARY STATISTICS")
print(f"{'=' * 80}")

print("\n🔍 Binary Detection:")
total_binary = sum(binary_stats.values())
for category, count in binary_stats.items():
    pct = (count / total_binary * 100) if total_binary > 0 else 0
    print(f"   {category.title()}: {count}/{total_binary} ({pct:.1f}%)")

print("\n📊 Classification Distribution:")
total_class = sum(classification_stats.values())
for level, count in classification_stats.items():
    pct = (count / total_class * 100) if total_class > 0 else 0
    icon = {"Safe": "🟢", "Moderate": "🟡", "High": "🔴"}.get(level, "❓")
    print(f"   {icon} {level}: {count}/{total_class} ({pct:.1f}%)")

# Flagged queries for review
if flagged_queries:
    print(f"\n🚨 FLAGGED FOR REVIEW ({len(flagged_queries)} queries):")
    print("─" * 60)
    seen = set()
    for idx, query, reason in flagged_queries:
        key = (idx, query)
        if key not in seen:
            print(f"   {idx}. \"{query}\" (Reason: {reason})")
            seen.add(key)
else:
    print("\n✅ No queries flagged for review!")

print(f"\n{'=' * 80}\n")
```

### Tutorial Summary

You've built a complete LLM-as-a-Judge system that:

1. ✅ Created an organized notebook structure
2. ✅ Built and validated binary and classification judges
3. ✅ Created a production system with automatic evaluations
4. ✅ Processed real queries with AI-generated responses
5. ✅ Automatically flagged problematic content for review

This system can scale to thousands of queries and provides detailed metrics for content moderation at scale.

---

## Error Handling

The SDK provides specific exception classes for different error scenarios.

### Exception Hierarchy

```python
from arato_client import (
    AratoAPIError,           # Base exception
    APIConnectionError,      # Connection issues
    BadRequestError,         # 400 errors (invalid input)
    AuthenticationError,     # 403 errors (invalid API key)
    NotFoundError,           # 404 errors (resource not found)
    InternalServerError      # 5xx errors (server issues)
)
```

### Basic Error Handling

```python
from arato_client import AratoClient, NotFoundError, BadRequestError

client = AratoClient()

try:
    notebook = client.notebooks.retrieve(notebook_id="invalid_id")
except NotFoundError as e:
    print(f"Notebook not found: {e.message}")
    print(f"Status code: {e.response.status_code}")
except BadRequestError as e:
    print(f"Invalid request: {e.message}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Comprehensive Error Handling

```python
from arato_client import (
    AratoClient,
    APIConnectionError,
    AuthenticationError,
    NotFoundError,
    BadRequestError,
    InternalServerError
)

client = AratoClient()

try:
    # Attempt operation
    notebook = client.notebooks.create(name="Test")
    
except APIConnectionError as e:
    print(f"❌ Connection error: {e.message}")
    print("   Please check your internet connection")
    
except AuthenticationError as e:
    print(f"❌ Authentication failed: {e.message}")
    print("   Please verify your API key")
    
except BadRequestError as e:
    print(f"❌ Invalid request: {e.message}")
    if e.response:
        try:
            error_details = e.response.json()
            print(f"   Details: {error_details}")
        except:
            pass
    
except NotFoundError as e:
    print(f"❌ Resource not found: {e.message}")
    
except InternalServerError as e:
    print(f"❌ Server error: {e.message}")
    print("   Please try again later or contact support")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

### Retrying Failed Requests

```python
import time
from arato_client import AratoClient, InternalServerError, APIConnectionError

client = AratoClient()

def create_notebook_with_retry(name, max_retries=3):
    """Create notebook with automatic retry on transient errors."""
    for attempt in range(max_retries):
        try:
            return client.notebooks.create(name=name)
        except (InternalServerError, APIConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⚠️  Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise  # Re-raise on final attempt

notebook = create_notebook_with_retry("My Notebook")
```

---

## Best Practices

### 1. Use Environment Variables for API Keys

**✅ Good:**
```python
import os
from arato_client import AratoClient

client = AratoClient()  # Uses ARATO_API_KEY from environment
```

**❌ Avoid:**
```python
# Never hardcode API keys
client = AratoClient(api_key="sk-1234567890...")  # Don't do this!
```

### 2. Use Context Managers

**✅ Good:**
```python
with AratoClient() as client:
    notebooks = client.notebooks.list()
    # Client automatically closes
```

**❌ Avoid:**
```python
client = AratoClient()
notebooks = client.notebooks.list()
# Client not properly closed
```

### 3. Poll Run Status Efficiently

**✅ Good:**
```python
import time

run = client.notebooks.experiments.runs.create(...)
run_id = run['id']

# Poll with exponential backoff
wait_time = 2
max_wait = 30

while True:
    run_details = client.notebooks.experiments.runs.retrieve(...)
    if run_details['status'] in ['done', 'failed']:
        break
    
    time.sleep(min(wait_time, max_wait))
    wait_time *= 1.5
```

**❌ Avoid:**
```python
# Too frequent polling
while True:
    run_details = client.notebooks.experiments.runs.retrieve(...)
    if run_details['status'] == 'done':
        break
    time.sleep(0.5)  # Polling every 500ms is too frequent
```

### 4. Handle Errors Gracefully

**✅ Good:**
```python
from arato_client import NotFoundError

try:
    notebook = client.notebooks.retrieve(notebook_id=notebook_id)
except NotFoundError:
    # Create if it doesn't exist
    notebook = client.notebooks.create(name="New Notebook")
```

### 5. Use Descriptive Names

**✅ Good:**
```python
notebook = client.notebooks.create(
    name=f"Customer Sentiment Analysis - {datetime.now().strftime('%Y-%m-%d')}",
    description="Analyzing customer feedback sentiment with GPT-4",
    tags=["sentiment", "customer-feedback", "production"]
)
```

**❌ Avoid:**
```python
notebook = client.notebooks.create(name="test")  # Too vague
```

### 6. Organize with Tags

```python
# Use consistent tagging for organization
notebook = client.notebooks.create(
    name="Q4 Marketing Campaign",
    tags=["marketing", "q4-2025", "production", "high-priority"]
)

dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Campaign Messages",
    tags=["marketing", "q4-2025", "test-data"]
)
```

### 7. Validate Data Before Creating Datasets

**✅ Good:**
```python
# Validate dataset structure
content = [
    {"input": "text1", "label": "positive"},
    {"input": "text2", "label": "negative"}
]

# Check all rows have required fields
required_fields = ["input", "label"]
for i, row in enumerate(content):
    missing = [f for f in required_fields if f not in row]
    if missing:
        raise ValueError(f"Row {i} missing fields: {missing}")

dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Validated Dataset",
    content=content
)
```

### 8. Use Appropriate Model Parameters

```python
# For consistent/deterministic outputs (judges, evaluations)
deterministic_config = {
    "model_id": "gpt-4o-mini",
    "vendor_id": "openai",
    "prompt_template": "...",
    "model_parameters": {
        "temperature": 0.1,  # Low temperature for consistency
        "max_tokens": 50
    }
}

# For creative outputs (content generation)
creative_config = {
    "model_id": "gpt-4o",
    "vendor_id": "openai",
    "prompt_template": "...",
    "model_parameters": {
        "temperature": 0.8,  # Higher temperature for creativity
        "max_tokens": 500
    }
}
```

### 9. Implement Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    notebook = client.notebooks.create(name="Test Notebook")
    logger.info(f"Created notebook: {notebook['id']}")
except Exception as e:
    logger.error(f"Failed to create notebook: {e}", exc_info=True)
```

### 10. Batch Operations Efficiently

**✅ Good:**
```python
# Create all datasets in one batch
datasets_to_create = [
    {"name": "Dataset 1", "content": [...]},
    {"name": "Dataset 2", "content": [...]},
    {"name": "Dataset 3", "content": [...]}
]

created_datasets = []
for ds_config in datasets_to_create:
    dataset = client.notebooks.datasets.create(
        notebook_id=notebook_id,
        **ds_config
    )
    created_datasets.append(dataset)
    
logger.info(f"Created {len(created_datasets)} datasets")
```

---

## API Reference

### Complete Method Signatures

#### Notebooks

```python
# List all notebooks
client.notebooks.list() -> Dict[str, List[Dict[str, Any]]]

# Create a notebook
client.notebooks.create(
    name: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None
) -> Dict[str, Any]

# Retrieve a notebook
client.notebooks.retrieve(
    notebook_id: str
) -> Dict[str, Any]
```

#### Global Datasets

```python
# List global datasets
client.datasets.list() -> Dict[str, List[Dict[str, Any]]]

# Create global dataset
client.datasets.create(
    name: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    content: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]

# Retrieve global dataset
client.datasets.retrieve(
    dataset_id: str
) -> Dict[str, Any]
```

#### Notebook Datasets

```python
# List notebook datasets
client.notebooks.datasets.list(
    notebook_id: str
) -> Dict[str, List[Dict[str, Any]]]

# Create notebook dataset
client.notebooks.datasets.create(
    notebook_id: str,
    name: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    content: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]

# Retrieve notebook dataset
client.notebooks.datasets.retrieve(
    notebook_id: str,
    dataset_id: str
) -> Dict[str, Any]
```

#### Experiments

```python
# List experiments
client.notebooks.experiments.list(
    notebook_id: str
) -> Dict[str, List[Dict[str, Any]]]

# Create experiment
client.notebooks.experiments.create(
    notebook_id: str,
    name: str,
    prompt_config: Dict[str, Any],
    description: Optional[str] = None,
    prompt_type: str = "generating_prompt",
    color_index: Optional[int] = None,
    dataset_id: Optional[str] = None
) -> Dict[str, Any]

# Retrieve experiment
client.notebooks.experiments.retrieve(
    notebook_id: str,
    experiment_id: str
) -> Dict[str, Any]

# Update experiment
client.notebooks.experiments.update(
    notebook_id: str,
    experiment_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    prompt_config: Optional[Dict[str, Any]] = None,
    color_index: Optional[int] = None,
    dataset_id: Optional[str] = None
) -> Dict[str, Any]
```

#### Runs

```python
# List runs
client.notebooks.experiments.runs.list(
    notebook_id: str,
    experiment_id: str
) -> Dict[str, List[Dict[str, Any]]]

# Create run
client.notebooks.experiments.runs.create(
    notebook_id: str,
    experiment_id: str,
    api_keys: Dict[str, str],
    callback_url: Optional[str] = None
) -> Dict[str, Any]

# Retrieve run
client.notebooks.experiments.runs.retrieve(
    notebook_id: str,
    experiment_id: str,
    run_id: str
) -> Dict[str, Any]
```

#### Evaluations

```python
# List evals
client.notebooks.experiments.evals.list(
    notebook_id: str,
    experiment_id: str
) -> Dict[str, List[Dict[str, Any]]]

# Create eval
client.notebooks.experiments.evals.create(
    notebook_id: str,
    experiment_id: str,
    name: str,
    eval_type: Literal["Numeric", "Binary", "Classification"],
    context: str = "prompt_and_response",
    prompt: Optional[str] = None,
    ranges: Optional[List[Dict[str, Any]]] = None,
    fail_on_positive: Optional[bool] = None,
    classes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]

# Retrieve eval
client.notebooks.experiments.evals.retrieve(
    notebook_id: str,
    experiment_id: str,
    eval_id: str
) -> Dict[str, Any]

# Update eval
client.notebooks.experiments.evals.update(
    notebook_id: str,
    experiment_id: str,
    eval_id: str,
    name: Optional[str] = None,
    context: Optional[str] = None,
    ranges: Optional[List[Dict[str, Any]]] = None,
    fail_on_positive: Optional[bool] = None
) -> Dict[str, Any]
```

---

## Additional Resources

- **API Specification**: See `sdk-api-spec.yaml` for complete OpenAPI documentation
- **Example Code**: Check `example.py` for usage examples
- **Source Code**: Available at https://github.com/AratoAi/arato-python-sdk
- **Support**: Contact support@arato.ai

---

## Changelog

### Version 1.0.6
- Initial public release
- Full support for notebooks, datasets, experiments, runs, and evaluations
- Synchronous and asynchronous client implementations
- Comprehensive error handling

---

## License

Proprietary - Copyright © 2025 Arato AI

---

*Last Updated: October 27, 2025*

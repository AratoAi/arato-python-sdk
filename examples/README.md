# Arato Python SDK Examples

This directory contains comprehensive examples demonstrating various features and best practices for using the Arato Python SDK.

## Prerequisites

Before running these examples, ensure you have:

1. **Python 3.7 or higher** installed
2. **Arato Python SDK** installed:
   ```bash
   pip install arato-client
   ```
3. **API Key** from Arato platform
4. **Environment setup** - Create a `.env` file in your project root:
   ```
   ARATO_API_KEY=your_api_key_here
   OPENAI_API_KEY=your_openai_key_here  # For LLM examples
   ```

## Available Examples

### 1. **quickstart.py** - Getting Started
**What it demonstrates:**
- Basic SDK initialization
- Creating notebooks and datasets
- Setting up experiments
- Running experiments and monitoring execution

**When to use:**
- First time using the SDK
- Understanding the basic workflow
- Quick reference for common operations

**Run it:**
```bash
python examples/quickstart.py
```

**Key concepts:**
- Client initialization with API key
- Resource creation (notebooks, datasets, experiments)
- Run execution and polling
- Basic error handling

---

### 2. **notebooks_and_datasets.py** - Data Management
**What it demonstrates:**
- CRUD operations for notebooks
- Creating global and notebook-scoped datasets
- Managing dataset content
- Querying and filtering resources

**When to use:**
- Managing multiple notebooks
- Working with different dataset types
- Understanding dataset scoping
- Organizing your experiments

**Run it:**
```bash
python examples/notebooks_and_datasets.py
```

**Key concepts:**
- Notebook management (create, list, retrieve, update, delete)
- Global datasets vs. notebook-scoped datasets
- Dataset content structure
- Resource listing and pagination

---

### 3. **experiments_and_runs.py** - Experiment Execution
**What it demonstrates:**
- Creating experiments with different configurations
- Running multiple experiments for comparison
- Analyzing experiment results
- Working with evaluations

**When to use:**
- A/B testing different prompts or parameters
- Comparing model performance
- Running systematic evaluations
- Analyzing token usage and costs

**Run it:**
```bash
python examples/experiments_and_runs.py
```

**Key concepts:**
- Experiment configuration (model, temperature, parameters)
- Run execution and monitoring
- Result analysis and comparison
- Creating and managing evaluations

---

### 4. **similarity_eval_example.py** - Similarity Evaluations
**What it demonstrates:**
- Creating Similarity evaluations
- Comparing AI outputs to expected results
- Using cosine and jaccard similarity algorithms
- Setting similarity thresholds for pass/fail criteria

**When to use:**
- Testing Q&A systems against expected answers
- Measuring semantic similarity of responses
- Validating AI output consistency
- Quality assurance for generated content

**Run it:**
```bash
python examples/similarity_eval_example.py
```

**Key concepts:**
- Similarity eval types (cosine, jaccard)
- Threshold configuration (0-100%)
- Comparing responses to dataset fields
- Semantic similarity scoring

**Note:** Verify your Arato API version supports Similarity evaluations.

---

### 5. **async_usage.py** - Asynchronous Operations
**What it demonstrates:**
- Using AsyncAratoClient for concurrent operations
- Batch creating resources in parallel
- Async/await patterns
- Error handling in async context

**When to use:**
- Processing large batches of data
- Improving performance with concurrent operations
- Building async applications
- Integrating with async frameworks

**Run it:**
```bash
python examples/async_usage.py
```

**Key concepts:**
- AsyncAratoClient initialization
- Concurrent resource creation with `asyncio.gather()`
- Async context managers
- Error handling in async functions
- Manual client lifecycle management

---

### 5. **error_handling.py** - Error Management
**What it demonstrates:**
- Different exception types in the SDK
- Proper error handling patterns
- Retry logic for transient errors
- Graceful degradation strategies
- Context-aware error messages

**When to use:**
- Building production applications
- Implementing robust error recovery
- Debugging API issues
- Understanding SDK exceptions

**Run it:**
```bash
python examples/error_handling.py
```

**Key concepts:**
- Exception hierarchy (BadRequestError, NotFoundError, etc.)
- Retry logic with exponential backoff
- Comprehensive try/except patterns
- Logging and debugging
- Error recovery workflows

---

### 6. **tutorials/llm_as_judge.ipynb** - Complete Use Case Tutorial
**What it demonstrates:**
- End-to-end LLM-as-a-Judge workflow
- Content moderation system
- Multiple evaluation approaches
- Advanced experiment patterns

**When to use:**
- Building content moderation systems
- Understanding complete workflows
- Learning advanced patterns
- Real-world application examples

**Run it:**
```bash
# Open in Jupyter or VS Code
jupyter notebook examples/tutorials/llm_as_judge.ipynb
```

**Key concepts:**
- Multi-step experiment workflows
- LLM-as-a-Judge evaluation pattern
- Binary, numeric, and classification evaluations
- Iterative prompt improvement
- Result analysis and visualization

---

## Running Examples

### Basic Usage
Each example can be run independently:

```bash
# From the project root
python examples/<example_name>.py
```

### With Custom Environment File
```bash
# Specify a different .env file
export ARATO_API_KEY=your_key_here
python examples/quickstart.py
```

### Recommended Learning Path

For beginners, we recommend following this order:

1. **quickstart.py** - Start here to understand basics
2. **notebooks_and_datasets.py** - Learn data management
3. **experiments_and_runs.py** - Master experiment execution
4. **error_handling.py** - Implement robust error handling
5. **async_usage.py** - Optimize with async operations
6. **tutorials/llm_as_judge.ipynb** - See complete real-world example

## Common Patterns

### Creating a Client
```python
from dotenv import load_dotenv
from arato_client import AratoClient

load_dotenv()
client = AratoClient()  # Uses ARATO_API_KEY from environment
```

### Creating a Notebook
```python
notebook = client.notebooks.create(
    name="My Notebook",
    description="Description here",
    tags=["tag1", "tag2"]
)
notebook_id = notebook['id']
```

### Creating a Dataset
```python
dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="My Dataset",
    content=[
        {"id": 1, "text": "Example 1"},
        {"id": 2, "text": "Example 2"}
    ]
)
```

### Running an Experiment
```python
# Create experiment with dataset
experiment = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="My Experiment",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": "Your prompt here",
        "model_parameters": {"temperature": 0.7}
    },
    dataset_id=dataset['id']
)

# Create and execute a run
run = client.notebooks.experiments.runs.create(
    notebook_id=notebook_id,
    experiment_id=experiment['id'],
    api_keys={"openai_api_key": "your_openai_key"}
)
```

## Error Handling

All examples include error handling. Here's a basic pattern:

```python
from arato_client import AratoAPIError, NotFoundError, BadRequestError

try:
    notebook = client.notebooks.retrieve(notebook_id="some_id")
except NotFoundError:
    print("Notebook not found")
except BadRequestError as e:
    print(f"Invalid request: {e.message}")
except AratoAPIError as e:
    print(f"API error: {e.message}")
```

## Environment Variables

All examples use these environment variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `ARATO_API_KEY` | Yes | Your Arato API key |
| `OPENAI_API_KEY` | For LLM examples | OpenAI API key for running experiments |

## Troubleshooting

### "AuthenticationError: Invalid API key"
- Check that `ARATO_API_KEY` is set in your `.env` file
- Verify your API key is valid in the Arato dashboard

### "Module not found: arato_client"
- Install the SDK: `pip install arato-client`
- Verify installation: `pip show arato-client`

### "APIConnectionError"
- Check your internet connection
- Verify the Arato API endpoint is accessible
- Check for proxy or firewall issues

### Rate Limiting
- Implement retry logic (see `error_handling.py`)
- Add delays between requests
- Use async operations for batch processing

## Additional Resources

- **Full Documentation**: See [DOCUMENTATION.md](../DOCUMENTATION.md)
- **API Reference**: Complete API details and parameters
- **GitHub Repository**: Report issues and contribute
- **Arato Dashboard**: Manage your API keys and view results

## Contributing

Found an issue or have an improvement? Please:
1. Check existing issues
2. Create a new issue with details
3. Submit a pull request with examples

## License

These examples are provided under the same license as the Arato Python SDK.

---

**Questions?** Check the [full documentation](../DOCUMENTATION.md) or reach out to support.

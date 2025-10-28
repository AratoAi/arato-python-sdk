"""
Error Handling Example - Arato Python SDK

This example demonstrates:
1. Different exception types and when they occur
2. Proper error handling patterns
3. Retry logic for transient errors
4. Graceful degradation strategies
5. Context-aware error messages
"""

import os
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from arato_client import (
    AratoClient,
    AratoAPIError,
    APIConnectionError,
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    InternalServerError
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize client
client = AratoClient()

# ============================================================================
# Part 1: Basic Error Handling
# ============================================================================

# Example 1: Handling NotFoundError
try:
    notebook = client.notebooks.retrieve(notebook_id="invalid_notebook_id")
except NotFoundError as e:
    print(f"NotFoundError caught: {e.message}")
except AratoAPIError as e:
    print(f"Unexpected API error: {e.message}")

# Example 2: Handling BadRequestError
try:
    notebook = client.notebooks.create(name="")
except BadRequestError as e:
    print(f"BadRequestError caught: {e.message}")
except AratoAPIError as e:
    print(f"Unexpected API error: {e.message}")

# Create a valid notebook for further examples
try:
    notebook = client.notebooks.create(
        name=f"Error Handling Demo - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        description="Demonstrating error handling patterns",
        tags=["error-handling", "demo"]
    )
    notebook_id = notebook['id']
    print(f"Created notebook: {notebook_id}")
except AratoAPIError as e:
    print(f"Failed to create notebook: {e.message}")
    notebook_id = None

# ============================================================================
# Part 2: Comprehensive Error Handling
# ============================================================================

def create_dataset_with_error_handling(client, nb_id, dataset_name, content):
    """Create a dataset with comprehensive error handling."""
    try:
        dataset = client.notebooks.datasets.create(
            notebook_id=nb_id,
            name=dataset_name,
            content=content
        )
        return dataset
        
    except APIConnectionError as e:
        logger.error("Connection error: %s", e.message)
        print("Network issue - please check your connection")
        return None
        
    except AuthenticationError as e:
        logger.error("Authentication failed: %s", e.message)
        print("Invalid API key or insufficient permissions")
        return None
        
    except BadRequestError as e:
        logger.error("Invalid request: %s", e.message)
        print(f"Invalid data provided: {e.message}")
        return None
        
    except NotFoundError as e:
        logger.error("Resource not found: %s", e.message)
        print("Notebook not found - it may have been deleted")
        return None
        
    except InternalServerError as e:
        logger.error("Server error: %s", e.message)
        print("Server error - please try again later")
        return None
        
    except AratoAPIError as e:
        logger.error("API error: %s", e.message)
        print(f"Unexpected API error: {e.message}")
        return None


if notebook_id:
    # Test with valid data
    dataset = create_dataset_with_error_handling(
        client=client,
        nb_id=notebook_id,
        dataset_name="Test Dataset",
        content=[{"id": 1, "value": "data1"}, {"id": 2, "value": "data2"}]
    )
    
    if dataset:
        print(f"Dataset created successfully: {dataset['id']}")
    
    # Test with invalid notebook ID
    dataset = create_dataset_with_error_handling(
        client=client,
        nb_id="invalid_id",
        dataset_name="Will Fail",
        content=[{"test": "data"}]
    )

# ============================================================================
# Part 3: Retry Logic for Transient Errors
# ============================================================================

def create_notebook_with_retry(client, name, max_retries=3, backoff_factor=2):
    """Create a notebook with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            nb = client.notebooks.create(name=name)
            logger.info("Successfully created notebook on attempt %d", attempt + 1)
            return nb
            
        except (InternalServerError, APIConnectionError) as e:
            # Retry on transient errors
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(
                    "Transient error on attempt %d: %s. Retrying in %ds...",
                    attempt + 1, e.message, wait_time
                )
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error("All %d attempts failed: %s", max_retries, e.message)
                print("All retry attempts exhausted")
                return None
                
        except (BadRequestError, AuthenticationError, NotFoundError) as e:
            # Don't retry on client errors
            logger.error("Non-retryable error: %s", e.message)
            print(f"Error that cannot be retried: {type(e).__name__}")
            return None
            
        except AratoAPIError as e:
            logger.error("API error: %s", e.message)
            print(f"API error: {e.message}")
            return None
    
    return None


# Test retry logic
notebook_with_retry = create_notebook_with_retry(
    client=client,
    name=f"Retry Test - {datetime.now().strftime('%H:%M:%S')}",
    max_retries=3
)

if notebook_with_retry:
    print(f"Notebook created after retries: {notebook_with_retry['id']}")

# ============================================================================
# Part 4: Graceful Degradation
# ============================================================================

def get_notebooks_with_fallback(client):
    """Retrieve notebooks with fallback to empty list on error."""
    try:
        response = client.notebooks.list()
        notebooks = response.get('data', [])
        logger.info("Retrieved %d notebooks", len(notebooks))
        return notebooks
        
    except AratoAPIError as e:
        logger.error("Failed to retrieve notebooks: %s", e.message)
        print("Could not fetch notebooks, using empty list")
        return []


notebooks = get_notebooks_with_fallback(client)
print(f"Retrieved {len(notebooks)} notebook(s)")

# ============================================================================
# Part 5: Context-Aware Error Messages
# ============================================================================

def create_experiment_with_context(client, nb_id, experiment_name):
    """Create an experiment with helpful, context-aware error messages."""
    try:
        experiment = client.notebooks.experiments.create(
            notebook_id=nb_id,
            name=experiment_name,
            prompt_config={
                "model_id": "gpt-4o-mini",
                "vendor_id": "openai",
                "prompt_template": "Test prompt",
                "model_parameters": {"temperature": 0.7}
            }
        )
        return experiment, None
        
    except NotFoundError:
        error_msg = (
            f"Notebook '{nb_id}' not found. "
            f"Please verify the notebook ID or create a new notebook first."
        )
        logger.error(error_msg)
        return None, error_msg
        
    except BadRequestError as e:
        error_msg = (
            f"Invalid experiment configuration for '{experiment_name}'. "
            f"Check your prompt_config and model parameters. "
            f"Details: {e.message}"
        )
        logger.error(error_msg)
        return None, error_msg
        
    except AuthenticationError:
        error_msg = (
            f"Authentication failed while creating experiment '{experiment_name}'. "
            f"Please check your API key and permissions."
        )
        logger.error(error_msg)
        return None, error_msg
        
    except AratoAPIError as e:
        error_msg = f"Failed to create experiment '{experiment_name}': {e.message}"
        logger.error(error_msg)
        return None, error_msg


if notebook_id:
    experiment, error = create_experiment_with_context(
        client=client,
        nb_id=notebook_id,
        experiment_name="Error Handling Test Experiment"
    )
    
    if experiment:
        print(f"Experiment created: {experiment['id']}")
    else:
        print(f"Failed to create experiment: {error}")

# ============================================================================
# Part 6: Error Recovery Workflow
# ============================================================================

def get_or_create_notebook(client, notebook_name):
    """Try to find an existing notebook by name, create it if not found."""
    try:
        # First, try to find existing notebook
        response = client.notebooks.list()
        notebooks = response.get('data', [])
        
        for nb in notebooks:
            if nb['name'] == notebook_name:
                logger.info("Found existing notebook: %s", nb['id'])
                print(f"Found existing notebook: {nb['id']}")
                return nb
        
        # Not found, create new one
        logger.info("Notebook '%s' not found, creating new one", notebook_name)
        print("Notebook not found, creating new one...")
        
        nb = client.notebooks.create(name=notebook_name)
        print(f"Created new notebook: {nb['id']}")
        return nb
        
    except AratoAPIError as e:
        logger.error("Failed to get or create notebook: %s", e.message)
        print(f"Error: {e.message}")
        return None


notebook = get_or_create_notebook(client, "Error Recovery Test Notebook")

print(f"\nError handling patterns demonstrated successfully")

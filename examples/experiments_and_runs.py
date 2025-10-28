"""
Experiments and Runs Example - Arato Python SDK

This example demonstrates:
1. Creating experiments with different configurations
2. Updating experiment prompts and parameters
3. Executing runs and tracking progress
4. Comparing results across multiple experiments
5. Using evaluations to assess outputs
"""

import os
import time
from datetime import datetime
from dotenv import load_dotenv
from arato_client import AratoClient

# Load environment variables
load_dotenv()

# Initialize client
client = AratoClient()

# ============================================================================
# Setup: Create Notebook and Dataset
# ============================================================================

# Create notebook
notebook = client.notebooks.create(
    name=f"Experiment Comparison - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    description="Comparing different prompt strategies and model parameters",
    tags=["experiments", "comparison", "demo"]
)
notebook_id = notebook['id']
print(f"Created notebook: {notebook_id}")

# Create dataset with test questions
dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Email Responses Dataset",
    description="Customer emails requiring responses",
    content=[
        {"email": "I haven't received my order yet. It's been 2 weeks!", "category": "shipping", "priority": "high"},
        {"email": "How do I reset my password?", "category": "technical", "priority": "medium"},
        {"email": "Your product is amazing! Just wanted to say thanks.", "category": "feedback", "priority": "low"},
        {"email": "I'd like to return my purchase. What's the process?", "category": "returns", "priority": "high"},
        {"email": "Do you offer bulk discounts for orders over 100 units?", "category": "sales", "priority": "medium"},
    ]
)
print(f"Created dataset with {len(dataset['content'])} emails")

# ============================================================================
# Part 1: Creating Different Experiments
# ============================================================================

# Experiment 1: Formal tone, low temperature
experiment_1 = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Formal Response Generator",
    description="Generates formal, professional email responses",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a professional customer service representative. "
            "Write a formal, courteous response to the following customer email.\n\n"
            "Customer Email: {{email}}\n"
            "Category: {{category}}\n"
            "Priority: {{priority}}\n\n"
            "Response:"
        ),
        "model_parameters": {"temperature": 0.3, "max_tokens": 200}
    },
    dataset_id=dataset['id'],
    color_index=0
)

# Experiment 2: Friendly tone, medium temperature
experiment_2 = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Friendly Response Generator",
    description="Generates warm, conversational email responses",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a friendly, helpful customer service rep. "
            "Write a warm, conversational response to this customer email. "
            "Use a casual but professional tone.\n\n"
            "Customer Email: {{email}}\n\n"
            "Response:"
        ),
        "model_parameters": {"temperature": 0.7, "max_tokens": 200}
    },
    dataset_id=dataset['id'],
    color_index=1
)

# Experiment 3: Concise responses
experiment_3 = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Concise Response Generator",
    description="Generates brief, to-the-point email responses",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "You are a customer service agent who values brevity. "
            "Write a short, direct response (2-3 sentences max) to this email.\n\n"
            "Email: {{email}}\n\n"
            "Brief Response:"
        ),
        "model_parameters": {"temperature": 0.5, "max_tokens": 100}
    },
    dataset_id=dataset['id'],
    color_index=2
)

print("Created 3 experiments: Formal, Friendly, Concise")

# ============================================================================
# Part 2: Running Experiments
# ============================================================================

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    print("Set OPENAI_API_KEY to execute runs")
else:
    experiments = [
        (experiment_1['id'], "Formal"),
        (experiment_2['id'], "Friendly"),
        (experiment_3['id'], "Concise")
    ]
    
    run_results = {}
    
    for exp_id, exp_name in experiments:
        # Create and start run
        run = client.notebooks.experiments.runs.create(
            notebook_id=notebook_id,
            experiment_id=exp_id,
            api_keys={"openai_api_key": openai_key}
        )
        
        # Wait for completion
        while True:
            run_details = client.notebooks.experiments.runs.retrieve(
                notebook_id=notebook_id,
                experiment_id=exp_id,
                run_id=run['id']
            )
            if run_details['status'] in ['done', 'failed']:
                break
            time.sleep(3)
        
        run_results[exp_name] = run_details
        print(f"Completed {exp_name} run: {run_details['status']}")
    
    # ========================================================================
    # Part 3: Comparing Results
    # ========================================================================
    
    print("\nComparing experiment results:")
    
    for exp_name, run_details in run_results.items():
        if run_details['status'] == 'done':
            content = run_details['content']
            total_tokens = sum(row.get('tokens_in', 0) + row.get('tokens_out', 0) for row in content)
            avg_response_length = sum(len(row.get('response', '')) for row in content) / len(content)
            
            print(f"\n{exp_name}:")
            print(f"  Total tokens: {total_tokens}")
            print(f"  Avg response length: {avg_response_length:.0f} chars")
            
            # Show one sample
            if content:
                print(f"  Sample: \"{content[0].get('response', '')[:80]}...\"")
    
    # ========================================================================
    # Part 4: Adding Evaluations
    # ========================================================================
    
    # Add a sentiment evaluation to the formal experiment
    eval_config = client.notebooks.experiments.evals.create(
        notebook_id=notebook_id,
        experiment_id=experiment_1['id'],
        name="Response Sentiment",
        eval_type="Classification",
        context="response",
        prompt="Classify the sentiment of this customer service response: Positive, Neutral, or Negative.",
        classes=[
            {"title": "Positive", "is_pass": True, "color": "green"},
            {"title": "Neutral", "is_pass": True, "color": "yellow"},
            {"title": "Negative", "is_pass": False, "color": "red"}
        ]
    )
    print("\nAdded sentiment evaluation to Formal experiment")
    
    # Update experiment with evaluation
    updated_experiment = client.notebooks.experiments.update(
        notebook_id=notebook_id,
        experiment_id=experiment_1['id'],
        name="Formal Response Generator (with Sentiment Analysis)"
    )

print(f"\nView notebook: https://app.arato.io/notebooks/{notebook_id}")

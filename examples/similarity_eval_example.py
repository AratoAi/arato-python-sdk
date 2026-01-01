"""
Similarity Evaluation Example - Arato Python SDK

This example demonstrates how to use Similarity evaluations to compare
AI-generated outputs against expected results using cosine or jaccard similarity.

Use case: Testing a question-answering system to ensure responses
match expected answers semantically.

NOTE: This example demonstrates the SDK support for Similarity evals.
      Verify that your Arato API version supports Similarity evaluations.
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
# Part 1: Setup - Create Notebook and Dataset
# ============================================================================

# Create notebook
notebook = client.notebooks.create(
    name=f"Similarity Eval Demo - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    description="Demonstrating similarity-based evaluation for Q&A systems",
    tags=["similarity", "evaluation", "demo"]
)
notebook_id = notebook['id']
print(f"✓ Created notebook: {notebook_id}")

# Create dataset with questions and expected answers
dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Q&A Test Dataset",
    description="Questions with expected answers for similarity testing",
    content=[
        {
            "user_query": "What is the capital of France?",
            "expected_result": "The capital of France is Paris."
        },
        {
            "user_query": "How many days are in a leap year?",
            "expected_result": "A leap year has 366 days."
        },
        {
            "user_query": "What is the largest planet in our solar system?",
            "expected_result": "Jupiter is the largest planet in our solar system."
        }
    ]
)
dataset_id = dataset['id']
print(f"✓ Created dataset with {len(dataset['content'])} Q&A pairs")

# ============================================================================
# Part 2: Create Experiment
# ============================================================================

experiment = client.notebooks.experiments.create(
    notebook_id=notebook_id,
    name="Q&A Assistant",
    description="Simple question-answering system",
    prompt_config={
        "model_id": "gpt-4o-mini",
        "vendor_id": "openai",
        "prompt_template": (
            "Answer the following question concisely and accurately:\n\n"
            "Question: {{user_query}}\n\n"
            "Answer:"
        ),
        "model_parameters": {
            "temperature": 0.3,
            "max_tokens": 100
        }
    },
    dataset_id=dataset_id,
    color_index=0
)
experiment_id = experiment['id']
print(f"✓ Created experiment: {experiment_id}")

# ============================================================================
# Part 3: Create Similarity Evaluation
# ============================================================================

print("\n📊 Creating Similarity Evaluation...")
print("This evaluation will:")
print("• Compare AI responses to expected_result field")
print("• Use cosine similarity algorithm")
print("• Require 75% similarity threshold to pass")

try:
    similarity_eval = client.notebooks.experiments.evals.create(
        notebook_id=notebook_id,
        experiment_id=experiment_id,
        name="Answer Similarity Check",
        eval_type="Similarity",
        context="response",  # Compare the AI's response
        threshold=75.0,  # Require 75% similarity to pass
        compare_to_field="expected_result"  # Compare against this field from dataset
    )
    eval_id = similarity_eval['id']
    print(f"✓ Created similarity eval: {eval_id}")
    print(f"  - Algorithm: cosine")
    print(f"  - Threshold: 75%")
    print(f"  - Comparing response to: expected_result")
    eval_created = True
except Exception as e:
    print(f"\n⚠️  Note: Similarity eval creation failed: {e}")
    print("    This may indicate the API doesn't support Similarity evals yet.")
    print("    The SDK is ready for when the API adds support.")
    print(f"\n    You can still view the notebook and experiment:")
    print(f"    https://app.arato.ai/notebooks/{notebook_id}")
    eval_created = False

# ============================================================================
# Part 4: Run the Experiment
# ============================================================================

openai_key = os.environ.get("OPENAI_API_KEY")

if not eval_created:
    print("\n⏭️  Skipping experiment run since eval wasn't created.")
    print(f"📝 Example code completed. Notebook: https://app.arato.ai/notebooks/{notebook_id}")
elif not openai_key:
    print("\n⚠️  Set OPENAI_API_KEY environment variable to execute the run")
    print(f"   View notebook: https://app.arato.ai/notebooks/{notebook_id}")
else:
    print("\n▶ Starting experiment run...")
    
    # Create and start run
    run = client.notebooks.experiments.runs.create(
        notebook_id=notebook_id,
        experiment_id=experiment_id,
        api_keys={"openai_api_key": openai_key}
    )
    run_id = run['id']
    print(f"  Run ID: {run_id}")
    
    # Wait for completion
    print("  Waiting for completion", end="", flush=True)
    while True:
        run_details = client.notebooks.experiments.runs.retrieve(
            notebook_id=notebook_id,
            experiment_id=experiment_id,
            run_id=run_id
        )
        
        status = run_details['status']
        if status in ['done', 'done_with_errors', 'failed']:
            break
        
        print(".", end="", flush=True)
        time.sleep(2)
    
    print(f"\n✓ Run completed with status: {status}")
    
    # ========================================================================
    # Part 5: Display Results with Similarity Scores
    # ========================================================================
    
    if status in ['done', 'done_with_errors']:
        print("\n" + "="*70)
        print("RESULTS - Similarity Evaluation")
        print("="*70)
        
        content = run_details.get('content', [])
        
        for i, row in enumerate(content, 1):
            print(f"\n📝 Question {i}: {row.get('user_query', '')}")
            print(f"Expected: {row.get('expected_result', '')}")
            print(f"Got:      {row.get('response', '')}")
            
            # Check if eval results exist
            evals = row.get('evals', [])
            if evals:
                for eval_result in evals:
                    if eval_result.get('name') == 'Answer Similarity Check':
                        score = eval_result.get('score', 0)
                        result = eval_result.get('result', 0)
                        
                        # Display similarity percentage
                        status_icon = "✅" if result == 1 else "❌"
                        print(f"{status_icon} Similarity: {score:.1f}% {'(PASS)' if result == 1 else '(FAIL)'}")
                        
                        # Show explanation if available
                        explanation = eval_result.get('explanation')
                        if explanation:
                            print(f"   {explanation}")
            else:
                print("⚠️  No evaluation results available yet")
        
        # Summary statistics
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        total_tests = len(content)
        passed_tests = 0
        total_similarity = 0
        
        for row in content:
            evals = row.get('evals', [])
            for eval_result in evals:
                if eval_result.get('name') == 'Answer Similarity Check':
                    passed_tests += eval_result.get('result', 0)
                    total_similarity += eval_result.get('score', 0)
        
        if total_tests > 0:
            pass_rate = (passed_tests / total_tests) * 100
            avg_similarity = total_similarity / total_tests
            
            print(f"Total questions: {total_tests}")
            print(f"Passed: {passed_tests} ({pass_rate:.1f}%)")
            print(f"Average similarity: {avg_similarity:.1f}%")
        
        print(f"\n🔗 View full results: https://app.arato.ai/notebooks/{notebook_id}")
    else:
        print(f"\n❌ Run failed. Check the notebook for details.")
        print(f"🔗 View notebook: https://app.arato.ai/notebooks/{notebook_id}")

print("\n" + "="*70)
print("Example completed!")
print("="*70)
print("\nKey takeaways:")
print("• Similarity eval compares AI outputs to expected results")
print("• Cosine similarity measures semantic similarity (0-100%)")
print("• Set a threshold to define pass/fail criteria")
print("• Alternative: Use 'jaccard' for token-based similarity")
print(f"\n🔗 Notebook: https://app.arato.ai/notebooks/{notebook_id}")

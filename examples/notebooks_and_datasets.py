"""
Notebooks and Datasets Example - Arato Python SDK

This example demonstrates:
1. Creating and listing notebooks
2. Creating global and notebook-scoped datasets
3. Retrieving and managing datasets
4. Working with dataset content
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from arato_client import AratoClient

# Load environment variables
load_dotenv()

# Initialize client
client = AratoClient()

# ============================================================================
# Part 1: Working with Notebooks
# ============================================================================

# List existing notebooks
notebooks_response = client.notebooks.list()
existing_notebooks = notebooks_response.get('data', [])
print(f"Found {len(existing_notebooks)} existing notebook(s)")

# Create a new notebook
notebook = client.notebooks.create(
    name=f"Data Management Demo - {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    description="Demonstrating notebook and dataset management capabilities",
    tags=["demo", "datasets", "tutorial"]
)
notebook_id = notebook['id']
print(f"Created notebook: {notebook_id}")

# Retrieve the notebook
retrieved_notebook = client.notebooks.retrieve(notebook_id=notebook_id)

# ============================================================================
# Part 2: Global Datasets
# ============================================================================

# Create a global dataset
global_dataset = client.datasets.create(
    name="Common Industry Terms",
    description="A reusable dataset of industry-specific terminology",
    tags=["global", "reference", "terminology"],
    content=[
        {"term": "API", "definition": "Application Programming Interface", "category": "technology"},
        {"term": "SDK", "definition": "Software Development Kit", "category": "technology"},
        {"term": "LLM", "definition": "Large Language Model", "category": "ai"},
        {"term": "CRUD", "definition": "Create, Read, Update, Delete", "category": "database"},
    ]
)
print(f"Created global dataset: {global_dataset['id']} ({len(global_dataset['content'])} rows)")

# List global datasets
global_datasets_response = client.datasets.list()
global_datasets = global_datasets_response.get('data', [])

# Retrieve a global dataset
retrieved_global = client.datasets.retrieve(dataset_id=global_dataset['id'])

# ============================================================================
# Part 3: Notebook-Scoped Datasets
# ============================================================================

# Create multiple datasets in the notebook
feedback_dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Customer Feedback",
    description="Customer product reviews and ratings",
    tags=["feedback", "reviews"],
    content=[
        {"customer_id": "C001", "product": "Widget Pro", "rating": 5, "comment": "Excellent product, exceeded expectations!", "sentiment": "positive"},
        {"customer_id": "C002", "product": "Widget Pro", "rating": 3, "comment": "Good but could be better", "sentiment": "neutral"},
        {"customer_id": "C003", "product": "Widget Lite", "rating": 2, "comment": "Disappointing quality", "sentiment": "negative"},
        {"customer_id": "C004", "product": "Widget Lite", "rating": 4, "comment": "Great value for money", "sentiment": "positive"},
    ]
)

tickets_dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Support Tickets",
    description="Customer support ticket data",
    tags=["support", "tickets"],
    content=[
        {"ticket_id": "T001", "subject": "Login issues", "priority": "high", "status": "open"},
        {"ticket_id": "T002", "subject": "Feature request", "priority": "low", "status": "closed"},
        {"ticket_id": "T003", "subject": "Billing question", "priority": "medium", "status": "in_progress"},
    ]
)

products_dataset = client.notebooks.datasets.create(
    notebook_id=notebook_id,
    name="Product Catalog",
    description="Available products and pricing",
    content=[
        {"sku": "WP-001", "name": "Widget Pro", "price": 99.99, "stock": 150},
        {"sku": "WL-001", "name": "Widget Lite", "price": 49.99, "stock": 300},
        {"sku": "WP-002", "name": "Widget Pro Max", "price": 149.99, "stock": 75},
    ]
)

# List all datasets in the notebook
notebook_datasets_response = client.notebooks.datasets.list(notebook_id=notebook_id)
notebook_datasets = notebook_datasets_response.get('data', [])
print(f"Created {len(notebook_datasets)} datasets in notebook")

# Retrieve a specific notebook dataset
retrieved_feedback = client.notebooks.datasets.retrieve(
    notebook_id=notebook_id,
    dataset_id=feedback_dataset['id']
)

# ============================================================================
# Part 4: Working with Dataset Content
# ============================================================================

# Analyze customer feedback dataset
feedback_data = retrieved_feedback['content']

# Calculate statistics
total_reviews = len(feedback_data)
avg_rating = sum(row['rating'] for row in feedback_data) / total_reviews
sentiment_counts = {}

for row in feedback_data:
    sentiment = row['sentiment']
    sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

print(f"Analyzed {total_reviews} reviews, avg rating: {avg_rating:.2f}")
print(f"Sentiment: {sentiment_counts}")

# Analyze support tickets dataset
tickets_data = tickets_dataset['content']

priority_counts = {}
status_counts = {}

for ticket in tickets_data:
    priority = ticket['priority']
    status = ticket['status']
    priority_counts[priority] = priority_counts.get(priority, 0) + 1
    status_counts[status] = status_counts.get(status, 0) + 1

print(f"Tickets by priority: {priority_counts}")
print(f"Tickets by status: {status_counts}")

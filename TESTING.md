# Arato Python SDK - Testing Guide

## Running Tests

### Install Test Dependencies

```bash
pip install -r requirements-test.txt
```

Or install the package with test dependencies:

```bash
pip install -e ".[test]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
# Test notebooks functionality
pytest tests/test_notebooks.py

# Test base resources
pytest tests/test_base.py

# Test client initialization
pytest tests/test_client.py
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_notebooks.py::TestNotebooksResource

# Run a specific test method
pytest tests/test_notebooks.py::TestNotebooksResource::test_delete_notebook
```

### Run with Coverage

```bash
# Generate coverage report
pytest --cov=arato_client --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Only Async Tests

```bash
pytest -m asyncio
```

### Run with Verbose Output

```bash
pytest -v
```

### Run with Output Capture Disabled

```bash
pytest -s
```

## Test Structure

- `tests/test_notebooks.py` - Tests for notebook resource operations (including delete)
- `tests/test_base.py` - Tests for base resource classes
- `tests/test_client.py` - Tests for client initialization and configuration

## Key Test Coverage

### Notebook Delete Functionality

The new delete functionality is thoroughly tested:

1. **Successful deletion** - Tests that DELETE requests are made correctly
2. **404 Not Found** - Tests error handling when notebook doesn't exist
3. **403 Forbidden** - Tests error handling when user lacks permissions
4. **Async deletion** - Tests asynchronous delete operations

### Base Resource Methods

All HTTP methods are tested:
- GET requests with and without parameters
- POST requests with payloads
- PUT requests for updates
- **DELETE requests** (new)

## Writing New Tests

When adding new features, follow these patterns:

```python
def test_new_feature(self, mock_client):
    """Test description."""
    # Arrange
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"expected": "data"}
    mock_client._request.return_value = mock_response
    
    # Act
    resource = ResourceClass(mock_client)
    result = resource.new_method()
    
    # Assert
    mock_client._request.assert_called_once()
    assert result["expected"] == "data"
```

For async tests:

```python
@pytest.mark.asyncio
async def test_new_async_feature(self, mock_async_client):
    """Test description."""
    # Arrange
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {"expected": "data"}
    
    async def mock_request(*args, **kwargs):
        return mock_response
    
    mock_async_client._request = mock_request
    
    # Act
    resource = AsyncResourceClass(mock_async_client)
    result = await resource.new_method()
    
    # Assert
    assert result["expected"] == "data"
```

## CI/CD Integration

To integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    pip install -r requirements-test.txt
    pytest --cov=arato_client --cov-report=xml
    
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

# Contributing to Drug Sales Prediction System

Thank you for your interest in contributing to the Drug Sales Prediction System! This document provides guidelines and information for contributors.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

## 🤝 Code of Conduct

This project adheres to a code of conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Focus on constructive feedback
- Accept responsibility for mistakes
- Show empathy towards other contributors
- Help create a positive community

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Quick Setup
```bash
# Fork and clone the repository
git clone https://github.com/Gihan007/drug-sales-prediction.git
cd drug-sales-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests to ensure everything works
pytest tests/
```

## 🛠️ Development Setup

### 1. Environment Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Verify setup
pre-commit run --all-files
```

### 2. IDE Configuration
We recommend using VS Code with the following extensions:
- Python (ms-python.python)
- Pylint (ms-python.pylint)
- Black Formatter (ms-python.black-formatter)
- isort (ms-python.isort)

### 3. Development Workflow
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code changes ...

# Run tests and linting
pytest tests/
black . && isort . && flake8 . && mypy src/

# Commit your changes
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

## 📝 Contributing Guidelines

### Code Style
- Follow **PEP 8** style guidelines
- Use **Black** for code formatting (line length: 88)
- Use **isort** for import sorting
- Add **type hints** to all functions
- Write **docstrings** following Google style

### Commit Messages
Follow conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Testing
- `chore`: Maintenance

Examples:
```
feat: add transfer learning for drug categories
fix: resolve shape mismatch in LSTM model
docs: update API documentation
test: add unit tests for meta-learning
```

### Code Quality Standards

#### Python Code
```python
# ✅ Good
def forecast_sales(category: str, date: str, model_type: str = "ensemble") -> tuple:
    """Forecast sales for a specific category and date.

    Args:
        category: Drug category (e.g., 'C1', 'C2')
        date: Target date in YYYY-MM-DD format
        model_type: Type of model to use

    Returns:
        Tuple of (forecast, date, plot_path, model_name)
    """
    # Implementation here
    pass

# ❌ Bad
def forecast_sales(category, date, model_type="ensemble"):
    # No type hints, no docstring
    pass
```

#### Testing
- Write unit tests for all new functions
- Aim for >90% code coverage
- Use descriptive test names
- Test edge cases and error conditions

```python
# ✅ Good test
def test_forecast_sales_valid_input():
    """Test forecasting with valid inputs."""
    result = forecast_sales("C1", "2024-12-01", "ensemble")
    assert len(result) == 4
    assert isinstance(result[0], (int, float))
    assert result[3] == "ensemble"

def test_forecast_sales_invalid_category():
    """Test forecasting with invalid category."""
    with pytest.raises(ValueError):
        forecast_sales("INVALID", "2024-12-01", "ensemble")
```

## 🔄 Pull Request Process

### 1. Before Submitting
- [ ] All tests pass (`pytest tests/`)
- [ ] Code is formatted (`black . && isort .`)
- [ ] Linting passes (`flake8 .`)
- [ ] Type checking passes (`mypy src/`)
- [ ] Documentation is updated
- [ ] Commit messages follow conventional format

### 2. PR Template
Please use the following template for pull requests:

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No breaking changes
```

### 3. Review Process
1. **Automated Checks**: CI/CD pipeline runs tests, linting, and type checking
2. **Code Review**: At least one maintainer reviews the code
3. **Approval**: PR is approved and merged
4. **Deployment**: Changes are automatically deployed

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests matching pattern
pytest tests/ -k "forecast"
```

### Writing Tests
- Place tests in `tests/` directory
- Name test files as `test_*.py`
- Use `pytest` framework
- Mock external dependencies
- Test both success and failure cases

### Performance Testing
```bash
# Run performance benchmarks
pytest tests/test_performance.py -v

# Profile specific functions
python -m cProfile -s time src/models/transformer_model.py
```

## 📚 Documentation

### Code Documentation
- Add docstrings to all public functions
- Update type hints
- Document complex algorithms
- Include usage examples

### Project Documentation
- Update README.md for new features
- Add API documentation in `docs/`
- Update changelog
- Create tutorials for complex features

### Documentation Standards
```python
def complex_algorithm(data: pd.DataFrame, params: dict) -> np.ndarray:
    """Perform complex forecasting algorithm.

    This function implements a sophisticated forecasting algorithm that
    combines multiple techniques for improved accuracy.

    Args:
        data: Historical sales data with columns ['date', 'sales', 'category']
        params: Dictionary of algorithm parameters
            - learning_rate: Learning rate for optimization (default: 0.01)
            - epochs: Number of training epochs (default: 100)
            - hidden_size: Size of hidden layers (default: 64)

    Returns:
        Numpy array of forecasted values

    Raises:
        ValueError: If input data is invalid
        RuntimeError: If training fails to converge

    Example:
        >>> data = pd.read_csv('sales_data.csv')
        >>> params = {'learning_rate': 0.001, 'epochs': 200}
        >>> forecast = complex_algorithm(data, params)
        >>> print(f"Forecast shape: {forecast.shape}")
    """
```

## 🐛 Issue Reporting

### Bug Reports
When reporting bugs, please include:
- **Description**: Clear description of the issue
- **Steps to Reproduce**: Step-by-step instructions
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, dependencies
- **Logs/Error Messages**: Any relevant output

### Feature Requests
For new features, please include:
- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Alternative approaches considered
- **Use Case**: Example usage scenario

### Issue Labels
- `bug`: Something isn't working
- `enhancement`: New feature or improvement
- `documentation`: Documentation updates
- `question`: Further information needed
- `wontfix`: Will not be implemented

## 🎯 Areas for Contribution

### High Priority
- [ ] Performance optimization for large datasets
- [ ] Additional ML models (Attention mechanisms, Graph Neural Networks)
- [ ] Real-time prediction API
- [ ] Mobile application development

### Medium Priority
- [ ] Multi-language support
- [ ] Advanced visualization features
- [ ] Integration with pharmacy management systems
- [ ] Automated model retraining pipelines

### Research Opportunities
- [ ] Novel meta-learning algorithms
- [ ] Uncertainty quantification improvements
- [ ] Cross-regional transfer learning
- [ ] Healthcare-specific feature engineering

## 📞 Getting Help

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: Report bugs and request features
- **Email**: Contact maintainers directly for sensitive matters
- **Slack/Discord**: Join our community channels

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in release notes
- Featured in research publications
- Invited to present at conferences

Thank you for contributing to the Drug Sales Prediction System! 🚀
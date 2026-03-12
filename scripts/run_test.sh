#!/bin/bash
# Local test runner script

set -e

echo "Running ML-KEM-512 Test Suite"
echo "=============================="

# Run unit tests
echo ""
echo "1. Running unit tests..."
pytest tests/unit/ -v

# Run integration tests
echo ""
echo "2. Running integration tests..."
pytest tests/integration/ -v

# Run NIST vector tests
echo ""
echo "3. Running NIST vector tests..."
pytest tests/vectors/ -v

# Generate coverage report
echo ""
echo "4. Generating coverage report..."
pytest tests/ --cov=ml_kem_512 --cov-report=html --cov-report=term

# Run linting
echo ""
echo "5. Running linters..."
black --check ml_kem_512/ tests/
flake8 ml_kem_512/ tests/
pylint ml_kem_512/

# Run type checking
echo ""
echo "6. Running type checker..."
mypy ml_kem_512/

echo ""
echo "=============================="
echo "All tests passed! ✓"
echo "Coverage report: htmlcov/index.html"
# Mystique Makefile - Standalone development and build system
# Python/Rust hybrid package using maturin

# Variables
PYTHON := python3
PIP := pip3
MATURIN := maturin
PYTEST := pytest
FLAKE8 := flake8
PACKAGE_NAME := mystique
SRC_DIR := src/mystique
TESTS_DIR := tests

# Build configuration
CARGO_TARGET_DIR ?= target

# Default target
.DEFAULT_GOAL := help

# PHONY targets
.PHONY: help install install-deps dev test lint check coverage \
        build build-dev sdist clean clean-rust distclean \
        check-rust check-maturin version

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Utility checks
check-rust: ## Check if Rust toolchain is installed
	@which rustc > /dev/null 2>&1 || \
		(echo "Error: Rust toolchain not found. Install from https://rustup.rs/" && exit 1)
	@echo "Rust toolchain: $$(rustc --version)"

check-maturin: ## Check if maturin is installed
	@which $(MATURIN) > /dev/null 2>&1 || \
		(echo "Error: maturin not found. Install with: pip install maturin" && exit 1)
	@echo "Maturin: $$($(MATURIN) --version)"

version: ## Display current package version
	@grep '^version = ' pyproject.toml | awk -F'"' '{print $$2}'

# Installation targets
install-deps: ## Install Python dependencies only
	@echo "Installing Python dependencies..."
	$(PIP) install colorful numpy pandas scikit-learn scipy
	$(PIP) install pytest pytest-cov flake8
	@echo "Dependencies installed"

install: check-rust check-maturin install-deps ## Install package for development (editable)
	@echo "Installing $(PACKAGE_NAME) in development mode..."
	$(MATURIN) develop
	@echo "Installation complete"

dev: install ## Alias for install

# Testing and quality
test: ## Run tests with coverage
	@echo "Running tests with coverage..."
	$(PYTEST) -v \
		--cov=mystique \
		--cov-report=term-missing \
		--cov-report=html \
		--cov-fail-under=0 \
		$(TESTS_DIR)/
	@echo "Coverage report: htmlcov/index.html"

lint: ## Run flake8 linter
	@echo "Running flake8..."
	$(FLAKE8) $(SRC_DIR) $(TESTS_DIR)/ --max-line-length=100

check: lint test ## Run both lint and test

coverage: test ## Generate HTML coverage report (alias for test)
	@echo "Opening coverage report..."
	@open htmlcov/index.html 2>/dev/null || xdg-open htmlcov/index.html 2>/dev/null || echo "Coverage report: htmlcov/index.html"

# Build targets
build: check-rust check-maturin ## Build release wheel
	@echo "Building release wheel..."
	$(MATURIN) build --release
	@ls -lh $(CARGO_TARGET_DIR)/wheels/ 2>/dev/null || echo "Wheels built in $(CARGO_TARGET_DIR)/wheels/"

build-dev: check-rust check-maturin ## Build debug wheel
	@echo "Building debug wheel..."
	$(MATURIN) build
	@echo "Debug build complete: $(CARGO_TARGET_DIR)/wheels/"

sdist: check-rust check-maturin ## Build source distribution
	@echo "Building source distribution..."
	$(MATURIN) build --release --sdist
	@echo "Source distribution: $(CARGO_TARGET_DIR)/wheels/"

# Cleanup targets
clean: ## Remove build artifacts and cache files
	@echo "Cleaning build artifacts..."
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf build/
	@rm -rf dist/
	@rm -rf $(CARGO_TARGET_DIR)/wheels/
	@rm -f MANIFEST
	@rm -rf docs/build/
	@rm -f .coverage
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .pytest_cache
	@echo "Clean complete"

clean-rust: ## Clean Rust build artifacts
	@echo "Cleaning Rust artifacts..."
	@cargo clean 2>/dev/null || echo "cargo not found, skipping Rust clean"
	@rm -rf $(CARGO_TARGET_DIR)
	@echo "Rust artifacts cleaned"

distclean: clean clean-rust ## Complete cleanup including virtual environments
	@echo "Complete cleanup..."
	@rm -rf venv/
	@rm -rf .venv/
	@echo "Distribution clean complete"

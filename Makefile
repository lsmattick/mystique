AWS_ACCOUNT_ID ?= 364189071156
DOMAIN ?= s1-packages
REPO ?= python
FORMAT ?= pypi
AWS_DEFAULT_REGION ?= us-west-2
PACKAGE_NAME ?= s1_mystique
PKG_DIR ?= $(CARGO_TARGET_DIR)/wheels/$(PACKAGE_NAME)*
KEEP_OLD ?= 1
VERSION_BASE := $(shell grep version pyproject.toml | awk -F '=' '{print $$2}' | sed "s/\"//g")
VERSION_SUFFIX ?= $(shell find Makefile src/mystique pyproject.toml -type f | grep -v _version | sort | xargs md5sum | md5sum | awk '{print $$1}')
VERSION := $(VERSION_BASE)+$(VERSION_SUFFIX)
# export these variables, so we can use them from tests
export AWS_ACCOUNT_ID DOMAIN REPO AWS_DEFAULT_REGION VERSION PACKAGE_NAME

all: lint test
s1_py_pkg: clean dist upload

test:
	@python -m pytest -v --cov=mystique --cov-config=.coveragerc --cov-report term-missing --cov-fail-under=0 --cov-report html .

lint:
	@flake8

install:
	@maturin build --release --sdist
	@pip install . -v

update_version: ## update version string based on content hash
	@sed -i "s/^version = \".*\"/version = \"$(VERSION)\"/" pyproject.toml

dist: update_version
	# replace the package name to avoid collision with pypi-published mystique package
	@sed -i "s/^name = \".*\"/name = \"$(PACKAGE_NAME)\"/" pyproject.toml
	# add cargo during build time, as the image doesn't have it
	@curl -sSLf https://sh.rustup.rs | bash -s -- -y
	@maturin build --release --sdist --zig

codeartifact_login:
	@/OpenMail/packages/python/login.sh

codeartifact_twine_login: ## Log into Code Artifact and set up for twine. Supply AWS_PROFILE=<profile> to use local credentials.
	@aws --region $(AWS_DEFAULT_REGION) codeartifact login --tool twine --domain $(DOMAIN) --domain-owner $(AWS_ACCOUNT_ID) --repository $(REPO)

upload: codeartifact_twine_login codeartifact_login ## Upload package to AWS Code Artifact the current hash is not yet present
	if ! $$(aws codeartifact list-package-versions --domain $(DOMAIN) --repository $(REPO) --format $(FORMAT) --package $(PACKAGE_NAME) --query 'versions[].version' --output text | grep -q $(VERSION)); then \
		twine upload --repository codeartifact $(PKG_DIR) --verbose; \
	fi

delete_old: codeartifact_login
	@aws codeartifact list-package-versions --domain $(DOMAIN) --repository $(REPO) --format $(FORMAT) --package $(PACKAGE_NAME) --sort-by PUBLISHED_TIME --query "versions[$(KEEP_OLD):].version" --output text | \
		xargs aws codeartifact delete-package-versions --domain $(DOMAIN) --repository $(REPO) --format $(FORMAT) --package $(PACKAGE_NAME) --versions

clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' | xargs rm -rf
	@rm -rf build/
	@rm -rf dist/
	@rm -f MANIFEST
	@rm -rf docs/build/
	@rm -f .coverage
	@rm -rf *.egg*
	@rm -rf htmlcov
	@rm -f $(PKG_DIR)

.PHONY: test lint install clean

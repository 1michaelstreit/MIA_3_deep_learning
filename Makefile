# Makefile — Assignment 3: Deep Learning for Medical Image Classification

VENV_DIR := .venv
UV       := uv

ifeq ($(OS),Windows_NT)
	PY := $(VENV_DIR)/Scripts/python.exe

	GREEN  :=
	YELLOW :=
	BLUE   :=
	NC     :=
else
	PY := $(VENV_DIR)/bin/python

	GREEN  := \033[0;32m
	YELLOW := \033[0;33m
	BLUE   := \033[0;34m
	NC     := \033[0m
endif

.PHONY: help setup baseline train evaluate test clean generate-baseline

help:
	@echo "$(BLUE)Assignment 3 — Deep Learning for Medical Image Classification$(NC)"
	@echo ""
	@echo "  $(GREEN)make setup$(NC)      — create virtual environment and install dependencies"
	@echo "  $(GREEN)make baseline$(NC)   — evaluate the pre-trained reference model on the test set"
	@echo "  $(GREEN)make train$(NC)      — train your model (model.py + config.yaml) and save it"
	@echo "  $(GREEN)make evaluate$(NC)   — evaluate your trained model and show your score"
	@echo "  $(GREEN)make test$(NC)       — run the grading test suite locally"
	@echo "  $(GREEN)make clean$(NC)      — remove cache files"
	@echo ""
	@echo "Typical workflow:"
	@echo "  1. make setup"
	@echo "  2. Edit model.py and config.yaml"
	@echo "  3. make train"
	@echo "  4. make evaluate"
	@echo "  5. git add models/submission_cnn.pt && git commit && git push"

# ── Setup ─────────────────────────────────────────────────────────────────────

setup:
	@command -v $(UV) >/dev/null 2>&1 || \
		(echo "$(YELLOW)uv not found. Install from https://docs.astral.sh/uv/$(NC)" && exit 1)

	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(UV) venv --python 3.12 $(VENV_DIR); \
		echo "$(GREEN)Virtual environment created at $(VENV_DIR)$(NC)"; \
	else \
		echo "$(YELLOW)Virtual environment already exists — skipping creation$(NC)"; \
	fi

	@$(UV) pip install --python $(PY) -r requirements.txt

	@echo "$(GREEN)Setup complete.$(NC)"

ifeq ($(OS),Windows_NT)
	@echo "Activate with: .venv\\Scripts\\Activate.ps1"
else
	@echo "Activate with: source $(VENV_DIR)/bin/activate"
endif

# ── Reference model ───────────────────────────────────────────────────────────

baseline:
	$(PY) evaluate.py --baseline

# ── Train your model ──────────────────────────────────────────────────────────

train:
	$(PY) train.py

# ── Evaluate your trained model ───────────────────────────────────────────────

evaluate:
	$(PY) evaluate.py

# ── Grading tests ─────────────────────────────────────────────────────────────

test:
	$(PY) -m pytest tests/ -v

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
ifeq ($(OS),Windows_NT)
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@if exist $(VENV_DIR) rmdir /s /q $(VENV_DIR)
else
	find . -name '__pycache__' -not -path './$(VENV_DIR)/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -not -path './$(VENV_DIR)/*' -delete 2>/dev/null || true
	rm -rf .pytest_cache
	rm -rf $(VENV_DIR)
endif
	@echo "$(GREEN)Clean done.$(NC)"

# ── Instructor only ───────────────────────────────────────────────────────────

generate-baseline:
	$(PY) instructor/generate_baseline.py
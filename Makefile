PYTHON := python3
MAIN_SCRIPT := main.py
CONFIG_DIR := config
TRAINING_CONFIG := $(CONFIG_DIR)/training_cfg.yaml
REPORT_UTIL := src/utils/report_generator.py

.PHONY: help learn report clean data

help:
	@echo "Commands avaialble:"
	@echo "  make learn [GPU=true]       - Run training model."
	@echo "                               IF GPU=true, uses GPU (if aval)."
	@echo "  make report                 - Generate report (mock data)"
	@echo "  make clean                  - Clean generated data (моделі, датасет, звіти)."

learn:
	@echo "--- Execute learn model ---"
ifeq ($(GPU),true)
	@echo "Using GPU for training model..."
	$(PYTHON) $(MAIN_SCRIPT) --gpu --config $(TRAINING_CONFIG)
else
	@echo "CPU usage or automatic device detection for learning.."
	$(PYTHON) $(MAIN_SCRIPT) --config $(TRAINING_CONFIG)
endif

report:
	@echo "--- Generate report ---"
	$(PYTHON) $(REPORT_UTIL) $(TRAINING_CONFIG)

clean:
	@echo "--- Removing artifacts of the project ---"
	@echo "Removing 'dataset' folder..."
	rm -rf dataset
	@echo "Removing 'models' folder..."
	rm -rf models
	@echo "Removing 'reports' folder..."
	rm -rf reports
	@echo "Clean up is done."
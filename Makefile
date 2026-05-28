# =============================================================
# GOP_cluster_redo —— Makefile
# 用法 (推荐, conda):
#   make install     从 environment.yml 建/更新 conda 环境
#   make run         跑完整 pipeline (默认 config/config.yaml)
#   make all         install + run
#   make clean       删除 outputs/ 里的产物
#   make distclean   再删除 conda 环境
#
# 用法 (回退, pip venv —— 不保证数值复现, 见 README "环境注意事项"):
#   make install-venv
#   make run-venv
# =============================================================

ENV_NAME ?= GOP_cluster_redo
CONFIG   ?= config/config.yaml

# conda 环境里的 python (--no-capture-output 让脚本里的 print 实时打印)
CONDA_PY := conda run -n $(ENV_NAME) --no-capture-output python

# 回退用 venv
VENV     := .venv
VENV_PY  := $(VENV)/bin/python
PYTHON   ?= python3

.PHONY: help check-conda install run all clean distclean install-venv run-venv

help:
	@echo "----- conda (推荐) -----"
	@echo "  make install     create/update conda env from environment.yml"
	@echo "  make run         run pipeline with CONFIG=$(CONFIG)"
	@echo "  make all         install + run"
	@echo "  make clean       remove outputs"
	@echo "  make distclean   clean + remove conda env"
	@echo ""
	@echo "----- pip venv (回退, 数值不保证复现) -----"
	@echo "  make install-venv  build .venv from requirements.txt"
	@echo "  make run-venv      run pipeline using .venv"

# 检查 conda 是否安装; 没装就给出友好提示并终止
check-conda:
	@command -v conda >/dev/null 2>&1 || { \
		echo ""; \
		echo "================================================================"; \
		echo "  ERROR: 没有检测到 conda"; \
		echo "================================================================"; \
		echo ""; \
		echo "  本项目推荐走 conda 路径, 因为 pip wheel 装的 numpy / scipy /"; \
		echo "  umap-learn 跟 conda-forge 链接的 BLAS 二进制不一样, 会让"; \
		echo "  HDBSCAN 输出从 7 个 cluster 漂到 6 个 (详见 README)。"; \
		echo ""; \
		echo "  请先安装 Miniconda (体积小, ~100 MB):"; \
		echo "    https://docs.conda.io/projects/miniconda/en/latest/"; \
		echo ""; \
		echo "  macOS 一行装法:"; \
		echo "    brew install --cask miniconda"; \
		echo "    conda init zsh        # 然后重开终端"; \
		echo ""; \
		echo "  装完后重新跑:"; \
		echo "    make all"; \
		echo ""; \
		echo "  如果不想装 conda, 可以走 pip 回退 (不保证 cluster 数复现):"; \
		echo "    make install-venv && make run-venv"; \
		echo "================================================================"; \
		echo ""; \
		exit 1; \
	}
	@echo "[check] conda found: $$(conda --version)"

install: check-conda
	@echo ">>> 用 conda 从 environment.yml 建/更新环境: $(ENV_NAME)"
	conda env update -n $(ENV_NAME) -f environment.yml --prune || \
	conda env create -n $(ENV_NAME) -f environment.yml

run: check-conda
	@conda env list | grep -q "^$(ENV_NAME) " || { \
		echo ""; \
		echo "ERROR: conda 环境 '$(ENV_NAME)' 不存在, 先跑 'make install'"; \
		echo ""; \
		exit 1; \
	}
	$(CONDA_PY) -m src.main --config $(CONFIG)

all: install run

clean:
	rm -rf outputs/*.csv outputs/*.txt outputs/*.md

distclean: clean
	@if [ "$$CONDA_DEFAULT_ENV" = "$(ENV_NAME)" ]; then \
		echo ""; \
		echo "ERROR: 你正在 '$(ENV_NAME)' 环境里, conda 不允许删自己。"; \
		echo "请先 deactivate 再重试:"; \
		echo "  conda deactivate"; \
		echo "  make distclean"; \
		echo ""; \
		exit 1; \
	fi
	conda env remove -n $(ENV_NAME) -y || true
	rm -rf $(VENV)

# ---- 回退方案: pip venv ----
$(VENV_PY):
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip

install-venv: $(VENV_PY)
	$(VENV)/bin/pip install -r requirements.txt

run-venv:
	$(VENV_PY) -m src.main --config $(CONFIG)

#!/usr/bin/env bash
set -euo pipefail
python -V
pip install -r requirements.txt
python artifact.py --runs 3 --n 20000 --dim 256 --topk 10 --csv bench.csv

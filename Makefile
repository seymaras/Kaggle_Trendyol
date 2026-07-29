.PHONY: install install-gpu notebook check kernel negatives mine train classical ranker test testlike-smoke testlike test-retrieval sub-hybrid

install:
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m ipykernel install --user --name=kaggle-trendyol --display-name="Python (Kaggle Trendyol)"

install-gpu:
	.venv/bin/pip install -r requirements-gpu.txt

notebook:
	.venv/bin/jupyter-lab

check:
	.venv/bin/python -c "import numpy, pandas, sklearn, matplotlib, pyarrow; print('Ortam hazır')"
	@test -f data/items.csv && test -f data/terms.csv && test -f data/training_pairs.csv && echo 'Yarışma verisi: tamam' || echo 'UYARI: data/ klasöründe CSV eksik'
	@test -f artifacts/train_with_negatives.csv && echo 'Negatif çıktısı: tamam' || echo 'BİLGİ: artifacts/train_with_negatives.csv henüz yok (negatif script çalıştırılmalı)'

negatives:
	USE_FULL_DATA=false POS_SAMPLE_SIZE=50000 .venv/bin/python src/trendyol_negative_sampling_kaggle.py

train:
	.venv/bin/python src/trendyol_train_baseline.py

mine:
	.venv/bin/python src/trendyol_query_hard_negative_mining.py --experiment-id query_mining_v2

classical:
	.venv/bin/python src/trendyol_train_classical_v2.py --train artifacts/experiments/query_mining_v2/train_query_negatives.parquet --experiment-id classical_v3

ranker:
	.venv/bin/python src/trendyol_train_catboost_ranker.py --mode train --train artifacts/train_testlike.parquet --experiment-id catboost_ranker_v1 --task-type GPU

test-retrieval:
	.venv/bin/python src/build_test_retrieval.py --output artifacts/test_retrieval.parquet

test:
	PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -v

smoke:
	.venv/bin/python src/trendyol_query_hard_negative_mining.py --max-terms 8 --catalog-size 3000 --top-k 50 --batch-size 4 --experiment-id smoke_query_mining_v2

testlike-smoke:
	.venv/bin/python src/build_testlike_training.py --sample-queries 2000

testlike:
	.venv/bin/python src/build_testlike_training.py

sub-hybrid:
	.venv/bin/python src/make_submission_from_probs.py --mode hybrid --positive-rate 0.23 --min-per-query 1

kernel: install

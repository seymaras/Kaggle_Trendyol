.PHONY: install notebook check kernel

install:
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m ipykernel install --user --name=kaggle-trendyol --display-name="Python (Kaggle Trendyol)"

notebook:
	.venv/bin/jupyter-lab

check:
	.venv/bin/python -c "import numpy, pandas, sklearn, matplotlib, pyarrow; print('Ortam hazır')"
	@test -f data/items.csv && test -f data/terms.csv && test -f data/training_pairs.csv && echo 'Yarışma verisi: tamam' || echo 'UYARI: data/ klasöründe CSV eksik'
	@test -f artifacts/train_with_negatives.csv && echo 'Negatif çıktısı: tamam' || echo 'BİLGİ: artifacts/train_with_negatives.csv henüz yok (negatif script çalıştırılmalı)'

kernel: install

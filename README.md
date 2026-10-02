rg -n -A 35 -B 5 'def load_ocr_cache|load_ocr_cache\(|ocr_cache' Antrag/Classification/ocr.py Antrag/Classification/config.py Antrag/Classification/pnc_features.py

rg -n -A 25 -B 5 'ocr_cache' Antrag/Classification/ocr.py Antrag/Classification/config.py Antrag/Classification/pnc_features.py

rg -n -C 3 'ocr_cache|build_ocr_cache' Antrag/Classification/pnc_inputs.py


ocr = load_ocr_cache(config.pnc_ocr_cache) if modality in {"text", "fusion"} else {}



git add \
Extraction/__init__.py \
Extraction/config.py \
Extraction/documents.py \
Extraction/extract.py \
Extraction/llm.py \
Extraction/pages.py \
Extraction/response_model.py \
Extraction/securegpt_client.py \
Extraction/README.md




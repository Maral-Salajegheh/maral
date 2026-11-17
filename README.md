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



cd /home/I060689/Projects/life-docai/Antrag/Classification

grep -n "import_assign_splits" pnc_data.py
grep -n "IGNORE_INDEX" config.py pnc_stage2.py
grep -n "return cache, enriched, features" pnc_stage2.py
grep -n "validation_report_write_text" data.py
grep -n '"--seed"' pnc_inputs.py
ls model.py models.py training.py 2>/dev/null



python -c "import life_docai_analytics_antrag; print(life_docai_analytics_antrag.__file__)"
python -m pip show life-docai




[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "life-docai-analytics"
version = "0.1.0"
description = "Analytics components for Life DocAI"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["."]
include = ["life_docai_analytics_antrag*"]



du -h --max-depth=4 packages/life-docai-analytics | sort -h | tail -20



git add \
packages/life-docai-analytics/pyproject.toml \
packages/life-docai-analytics/life_docai_analytics_antrag/__init__.py \
packages/life-docai-analytics/life_docai_analytics_antrag/README.md \
packages/life-docai-analytics/life_docai_analytics_antrag/classification/**/*.py \
packages/life-docai-analytics/life_docai_analytics_antrag/classification/**/README.md \
packages/life-docai-analytics/life_docai_analytics_antrag/datasets/**/*.py \
packages/life-docai-analytics/life_docai_analytics_antrag/datasets/**/README.md




python -c '
import csv
from pathlib import Path

folder = Path.home() / "Projects/life-docai/packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD"
files = sorted(folder.glob("*predictions*.csv"))

if not files:
    raise SystemExit(f"No prediction CSV found in: {folder}")

for path in files:
    fp = 0
    predicted_mad = 0
    print(f"\nFile: {path.name}")

    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["predicted_class"].strip() == "MAD":
                predicted_mad += 1
                if row["page_class"].strip() != "MAD":
                    fp += 1
                    if fp <= 10:
                        print(
                            "False positive:",
                            row.get("image_path") or row.get("page_identity"),
                            "| Actual:", row["page_class"]
                        )

    print("Predicted MAD:", predicted_mad)
    print("MAD false positives:", fp)
    if predicted_mad:
        print("MAD precision: {:.2%}".format(
            (predicted_mad - fp) / predicted_mad
        ))
'




python -c '
import json
from pathlib import Path

folder = Path.home() / "Projects/life-docai/packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD"
found = False

for path in sorted(folder.glob("*.json")):
    with path.open(encoding="utf-8-sig") as f:
        data = json.load(f)

    sections = [("evaluation", data)]
    sections += [
        (split, data[split])
        for split in ("train", "val", "test")
        if isinstance(data.get(split), dict)
    ]

    for split, section in sections:
        for key in ("first_page", "segmentation"):
            metrics = section.get(key)
            if not isinstance(metrics, dict):
                continue
            found = True
            print(f"\nFile: {path.name}")
            print(f"Section: {split} | Metric: {key}")
            for name in ("precision", "recall", "f1", "accuracy", "support"):
                if name in metrics:
                    print(f"  {name}: {metrics[name]}")

if not found:
    print("No first_page or segmentation metrics found.")
'






python - <<'PY'
import csv
from collections import Counter
from pathlib import Path

root = Path.home() / "Projects/life-docai/packages/life-docai-analytics/life_docai_analytics_antrag/classification"
batch = "antrag_life_A00_G07_MAD"
data_dir = root / "output" / batch
results_dir = root / "results" / batch

with (data_dir / "stack_splits.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    test_stacks = {
        r["stack_id"]
        for r in csv.DictReader(f)
        if r["split"] == "test"
    }

files = sorted(results_dir.glob("predictions*.csv"))
if not files:
    raise SystemExit(f"No prediction files found: {results_dir}")

fields = [
    "stack_id", "masterindex_id", "process_id", "doc_id",
    "subdoc_idx", "image_id", "stack_position", "pdf_page_number",
    "sfdoc_class", "page_class", "predicted_class",
    "training_label_quality", "image_path",
]

for path in files:
    errors = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if (
                r["stack_id"] in test_stacks
                and r["page_class"].strip() != "MAD"
                and r["predicted_class"].strip() == "MAD"
            ):
                errors[r["page_identity"]] = r

    output = path.with_name(path.stem + "_MAD_FP_test.csv")
    counts = Counter()
    matched = set()

    with (data_dir / "stack_pages.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source, output.open("x", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()

        for r in csv.DictReader(source):
            identity = r["page_identity"]
            if identity not in errors:
                continue

            prediction = errors[identity]
            record = {key: r.get(key, "") for key in fields}
            record["page_class"] = prediction["page_class"]
            record["predicted_class"] = prediction["predicted_class"]
            writer.writerow(record)
            counts[prediction["page_class"]] += 1
            matched.add(identity)

    print(f"\nPrediction: {path.name}")
    print(f"MAD false positives in test: {len(errors)}")
    print(f"Matched pages: {len(matched)}")
    print(f"Actual classes: {dict(counts)}")
    print(f"Saved: {output}")
PY




cd ~/Projects/life-docai

pixi run -e dev python - <<'PY'
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

root = Path("packages/life-docai-analytics/life_docai_analytics_antrag/classification")
batch = "antrag_life_A00_G07_MAD"
data_dir = root / "output" / batch
results_dir = root / "results" / batch

prediction = results_dir / "predictions_final_model_final_fusion_mapping_v2_best_epoch_000.csv"

with (data_dir / "stack_splits.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    test_stacks = {
        r["stack_id"] for r in csv.DictReader(f)
        if r["split"] == "test"
    }

errors = {}
predicted_mad = 0

with prediction.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["stack_id"] not in test_stacks:
            continue
        if r["predicted_class"].strip() == "MAD":
            predicted_mad += 1
            if r["page_class"].strip() != "MAD":
                errors[r["page_identity"]] = r

fields = [
    "stack_id", "masterindex_id", "process_id", "doc_id",
    "subdoc_idx", "image_id", "stack_position", "pdf_page_number",
    "sfdoc_class", "page_class", "predicted_class",
    "training_label_quality", "image_path",
]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = prediction.with_name(
    prediction.stem + f"_MAD_FP_test_{stamp}.csv"
)

counts = Counter()
matched = set()

with (data_dir / "stack_pages.csv").open(
    newline="", encoding="utf-8-sig"
) as source, output.open(
    "x", newline="", encoding="utf-8-sig"
) as target:
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()

    for r in csv.DictReader(source):
        identity = r["page_identity"]
        if identity not in errors:
            continue
        pred = errors[identity]
        record = {key: r.get(key, "") for key in fields}
        record["page_class"] = pred["page_class"]
        record["predicted_class"] = pred["predicted_class"]
        writer.writerow(record)
        counts[pred["page_class"]] += 1
        matched.add(identity)

print("Prediction:", prediction.name)
print("Predicted MAD in test:", predicted_mad)
print("MAD false positives in test:", len(errors))
print("Matched pages:", len(matched))
print("Actual classes:", dict(counts))
if predicted_mad:
    print("MAD precision: {:.2%}".format(
        (predicted_mad - len(errors)) / predicted_mad
    ))
print("Saved:", output.resolve())
PY





python3 -c '
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in ["analyse_db_rows", "stacks_total", "stacks_used", "pages_used",
          "pages_excluded", "documents", "pages_on_fallback_key",
          "first_page_labels_changed_vs_mapping",
          "rendered_pages_without_row_in_used_mids"]:
    print(f"{k:42s} {str(a.get(k)):>10} -> {str(b.get(k)):>10}")
for section in ["page_class_counts", "stacks_excluded"]:
    print(f"\n{section}")
    x, y = a.get(section, {}), b.get(section, {})
    for key in sorted(set(x) | set(y)):
        print(f"  {key:28s} {x.get(key,0):>9} -> {y.get(key,0):>9}")
' classification/output/antrag_life_A00_G07_MAD/stack_population_report_v1.json classification/output/antrag_life_A00_G07_MAD/stack_population_report.json








##fals negative mad
cd ~/Projects/life-docai

pixi run -e dev python - <<'PY'
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

root = Path("packages/life-docai-analytics/life_docai_analytics_antrag/classification")
batch = "antrag_life_A00_G07_MAD"
data_dir = root / "output" / batch
results_dir = root / "results" / batch

prediction = results_dir / "predictions_final_model_final_fusion_mapping_v2_best_epoch_000.csv"

with (data_dir / "stack_splits.csv").open(
    newline="", encoding="utf-8-sig"
) as f:
    test_stacks = {
        r["stack_id"] for r in csv.DictReader(f)
        if r["split"] == "test"
    }

errors = {}
actual_mad = 0
true_positive = 0

with prediction.open(newline="", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["stack_id"] not in test_stacks:
            continue
        if r["page_class"].strip() == "MAD":
            actual_mad += 1
            if r["predicted_class"].strip() == "MAD":
                true_positive += 1
            else:
                identity = r["page_identity"]
                if identity in errors:
                    raise ValueError(f"Duplicate page_identity: {identity}")
                errors[identity] = r

fields = [
    "stack_id", "masterindex_id", "process_id", "doc_id",
    "subdoc_idx", "image_id", "stack_position", "pdf_page_number",
    "sfdoc_class", "page_class", "predicted_class",
    "training_label_quality", "image_path",
]

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output = prediction.with_name(
    prediction.stem + f"_MAD_FN_test_{stamp}.csv"
)

counts = Counter(r["predicted_class"].strip() for r in errors.values())
matched = set()

with (data_dir / "stack_pages.csv").open(
    newline="", encoding="utf-8-sig"
) as source, output.open(
    "x", newline="", encoding="utf-8-sig"
) as target:
    writer = csv.DictWriter(target, fieldnames=fields)
    writer.writeheader()

    for r in csv.DictReader(source):
        identity = r["page_identity"]
        if identity not in errors:
            continue
        pred = errors[identity]
        record = {key: r.get(key, "") for key in fields}
        record["page_class"] = pred["page_class"]
        record["predicted_class"] = pred["predicted_class"]
        writer.writerow(record)
        matched.add(identity)

print("Prediction:", prediction.name)
print("Actual MAD in test:", actual_mad)
print("Correctly predicted MAD:", true_positive)
print("MAD false negatives in test:", len(errors))
print("Predicted instead as:", dict(counts))
print("Matched pages:", len(matched))
if actual_mad:
    print("MAD recall: {:.2%}".format(true_positive / actual_mad))
if len(matched) != len(errors):
    print("WARNING: Some pages were not found in stack_pages.csv.")
print("Saved:", output.resolve())
PY






cd ~/Projects/life-docai && mkdir -p packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD/tb_check && pixi run -e dev python packages/life-docai-analytics/life_docai_analytics_antrag/classification/analysis/tb_datamatrix_check.py --pages-csv packages/life-docai-analytics/life_docai_analytics_antrag/classification/output/antrag_life_A00_G07_MAD/stack_pages.csv --render-root packages/life-docai-analytics/life_docai_analytics_antrag/datasets/aws_download_ingestion --limit-docs 300 --payload-dir packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD/tb_check/payloads --output packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD/tb_check/tb_check_labels.csv


check before run
cd ~/Projects/life-docai && pixi run -e dev python -c "import zxingcpp; print(zxingcpp.__version__)"


if error add this
cd ~/Projects/life-docai && pixi add -e dev --pypi zxing-cpp


pixi run -e dev python -c "import zxingcpp; from importlib.metadata import version; print(version('zxing-cpp'))"


pixi run -e dev python -m life_docai_analytics_antrag.classification.analysis.tb_datamatrix_check --batch-id antrag_life_A00_G07_MAD --limit-docs 300
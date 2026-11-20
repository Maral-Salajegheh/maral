# Technikblatt (TB) and the new SSTs: findings as of 20 November 2026

> This file contains only things that were observed in data or run. Assumptions are marked **(assumption)**.

---

## 1. Goal

When a new document arrives, the pipeline must determine its SST. The current training data contains only A00, G07 and MAD (TB0 is inside A00). The other SSTs (AME, AMD, AMB, AMU, AMW, AOR, E00, A01, LNS) differ from A00 mostly in **process**, not in **content**.

- The SST is a **document** label, not a stack label.
- In ANDB, `sfdoc_class` of these SSTs is mostly `Bestand/Default` (class 40).

---

## 2. SST rules (confirmed by Nina)

| Condition | SST |
|---|---|
| Normal + TB broken | AME |
| DÄV + TB broken | AMD |
| bAV (always, TB intact or broken) | AMB |
| U-Kasse (always, TB intact or broken) | AMU |
| Rahmenvertrag | AMW, **struck out, still open** |
| Duplicate | AOR |
| Reminder email | E00 |
| Supplementary document | A01 |
| Must be rescanned | LNS |

**Confirmed in data:** `MIH735396335` (AMB) has a complete, intact TB (`mid_broken=False`) and is still AMB. So "TB intact" alone does **not** mean A00. bAV must be checked first.

---

## 3. Structure of the Technikblatt

### Two different layouts

1. **Standard AXA TB**: heading "!! Technikblatt – zwingend mit dem Antrag einreichen !!", with Data Matrix codes.
2. **TB from the bAV / U-Kasse offer tool**: coloured bars, heading "5. Technikblatt" or "Technikblatt", titles such as "Angestelltenversorgung als arbeitgeberfinanzierte Unterstützungskasse". This layout has **no Data Matrix**. Seen in AMU.

### Pages of the standard TB

| Page | Data Matrix |
|---|---|
| Text pages (usually 2, sometimes 3) | 1 `AXA01...` code (+ 1 `NICHTGWGIVRELEVANT` code if the GwG flag is set) |
| Grid pages | 1 `AXA01...` code + up to 12 `AXA03` codes = **13** (**14** with GwG) |
| Last grid page | 1 `AXA01...` code + remaining chunks |

Examples from documents with `match=True`:

```
1 1 13 13 10          ← 2 text pages, 3 grid pages
2 2 14 14 7           ← with GwG flag
2 2 2 14 14 14 9      ← 3 text pages + GwG
1 0 1 13 13 13 8      ← one page code not read, still match=True
```

- Text pages carry **no AXA03 data**. All containers are on the grid pages.
- Text page 1 **prints** the key information: `Produkt:`, `Schicht`, `Gesellschaft:`, `Routing:`, `Durchführungsweg:`, `Finanzierungsart:`, `Zugangskanal:`, `Zugangsweg:`, `VN Nachname:`, etc.
- Some documents contain **only the TB text pages** and no grid pages. These yield no payload.

---

## 4. Data Matrix formats

### `AXA03` (data)

```
AXA03 | container (10) | chunk number (3) | total chunks (3) | Base64 ...
```

Chunks are concatenated in number order, then Base64-decoded, then gunzipped.

| Container | Meaning | Required? |
|---|---|---|
| `0000000100` | Partner | optional |
| `0000000101` | Zusatz | optional |
| `0000000102` | Antrag | **required** |
| `0000000103` | Angebot | **required** |
| `0000010110` | unknown | seen in 24 of 300 documents, no effect on the decision |

### `AXA01` (form identifier)

```
AXA01 + form number + (version) + P<form page number>
example: AXA01D21006844_(01.23)P03
```

- Present on **all AXA forms**, not only the TB. It is the small Data Matrix at the bottom or top of the page.
- TB forms: `D21011325`, `D21011326`, `D21012001`
- Other forms seen in AMB: `D21006836`, `D21012882`, `D21006947`, `D21006844`, `D21012001`
- **Does not contain the Antrag number or barcode.** It cannot be used to group the pages of one TB in production.
- It does give the **form type**, which is a potential deterministic signal for bAV (section 9).

### `NICHTGWGIVRELEVANT`

A GwG flag. In some documents it is repeated on every TB page.

---

## 5. AXA decision from the containers (`tbCheckDocClassForContainer`)

```
102 or 103 incomplete  → Bestand   ("Datamatrix fehlt / defekt, bitte als Bestand (AME,AMD,AMB,...) weitergeben")
Zusatz (101) missing   → Antrag/Technikblatt Partner und Antragsdaten
Partner (100) missing  → Antrag/Technikblatt Partnerdaten
all present            → Antrag/Technikblatt
```

---

## 6. Payload content

### Container 103 (Angebot)

- XML: `soProposalCalculateCustomerRequest` (namespace `xmlns.fja.com`), tags `pcInFilterDef:...`
- Key fields:

| Field | A00 (DÄV BU example) | AMB (`MIH735396335`, `MIH792896703`) | Meaning |
|---|---|---|---|
| `pdId` / `tfId` | `DLVSBV_3_2025` | (Rente, Schicht 2) | product code |
| `c_schicht` | `SCHICHT3` | `SCHICHT2` | pension layer |
| `kzbAV` | **absent** | **`DIREKT`** | bAV + Durchführungsweg |
| `kzFinanz` | `NICHTRELEVANT` | `AGANFINANZIERT` | employer contribution |
| `mandantId` | `DAEV` | `ALV` | AXA Leben or DÄV |
| `c_gesellschaft` | `DAEV` | `AXA` | |
| `c_zugangsweg` | `AXA_ZUGANGSWEG_1044` | `AXA_ZUGANGSWEG_2000` | |
| `vertriebsweg` | `AXA_KEINE_ANGABE` | `AXA_KEINE_ANGABE` | |

- Most other differences in the `diff` come from the **product type** (BU vs. Rente), not from bAV. For example, `berufsgruppe` and `risikogruppe` exist only for BU, and `fondsId`, `garantieniveau` and `kzRente` only for Rente.
- Product code pattern: **the middle number is the Schicht**: `DLVSBV-3-2025`, `ALVFRV-3-2025` (A00); `ALVSBV-2-2025`, `ALVRX-2-2025` (AMB/AMU).

### Container 102 (Antrag)

- XML: `soProcessApplicationImportRequest`, tags `applicationFilterDef:...` and `lifeContract:...`
- `lifeContract:companyId` = `ALV`
- A search for bAV / Durchführungsweg / Routing / Finanz terms returned **nothing**. For bAV, **103 is the main source**.

### A00 sample (201 files of container 103, from the older run with `--payload-dir`)

| Field | A00 (201 files) | AMB (2 documents) |
|---|---|---|
| `kzbAV` | **never** | `DIREKT` |
| `c_schicht` | `SCHICHT3` (195), `SCHICHT1` (6), **no `SCHICHT2`** | `SCHICHT2` |
| `kzFinanz` | `NICHTRELEVANT` (115), absent in 86 files | `AGANFINANZIERT` |

- In this sample, all three fields separate A00 from AMB.
- **`kzbAV` is the main signal.** It never appears in A00, and conceptually it means "bAV" directly.
- **`kzFinanz` is only meaningful when present.** It is absent in 86 files, probably depending on the product type (e.g. BU). So "no `kzFinanz`" does **not** mean "not bAV".
- **`SCHICHT2` alone is not enough.** There was no Riester in this sample, so it is unknown how Riester is recorded. Use `c_schicht` only as a confirmation.
- The sample is small: about 200 A00 and 2 AMB.
- Technical note: without `<` at the start of the `grep` pattern, closing tags are counted too and lines without values appear in the output.

### Current rule for a complete TB

```
kzbAV present    → bAV  (AMB; not yet confirmed for AMU)
kzbAV absent     → not bAV → TB check (AME / AMD / A00)
confirmation:     c_schicht = SCHICHT2  and  kzFinanz = AGANFINANZIERT
```

---

## 7. Decoder validation on the A00/G07/MAD batch

### Run 1 (grouping by `segmentation_target`)

- 300 documents, 97% match, 9 mismatches:
  - mapping error (all pages without Data Matrix): `36045`, `mkkvb…297879`
  - decode failure: `20375`, `50078`, `27665`, `16770`, `55048`
  - TB split by interleaved `image_id` order: `37388`
  - we read more than AXA (harmless): `48136`

### Run 2 (grouping by `doc_id` + fallback)

- **297 of 300 correct**, 3 mismatches:
  - mapping error: `mkkvb0300000297879`, `mklia0000000027334`
  - decode failure: `mklia0000000020375` (`0 0 12 12 7`, `missing 102:1`, Partner never seen)
- Containers: 102 and 103 in 298 documents, 101 in 186, 100 in 185, `0000010110` in 24.
- Because the grouping changed, the 300-document sample is not the same as in run 1.

### Analysis of `20375`

- `0 0` on pages 1–2 is harmless, since those pages only have `AXA01`.
- The real problem is on the grid pages: one code missing on each, Partner missing entirely, and one chunk of 102 missing.
- **(assumption)** scan quality. Pages 3–5 need to be inspected.

### Lessons

- The real page order is `image_id` (checked in ANDB). Documents can be interleaved (e.g. `37388`). For validation, the mapping's `doc_id` is correct. In production there is no `doc_id`.
- `libdmtx` is not available on the server (no `apt-get`, and conda-forge has no `libdmtx`). It was removed from the fallback.
- `zlib.error` was missing from the `except` clause and crashed the run. It was added, along with the `conflicting_chunks` column (same chunk number with two different texts, i.e. two TBs mixed together).

---

## 8. AMB / AMU findings (separate render, 300 DPI)

- 734 images, 115 with a Data Matrix.
- MIDs were resolved correctly (mapping was fine).
- Most AMB documents **have no grid pages** (`axa03_chunk_count = 0`), so they have no payload.
- 4 MIDs had complete containers. Two have a complete 103 and **both are AMB**:
  - `MIH735396335`: `kzbAV=DIREKT`, `SCHICHT2`, `AGANFINANZIERT`, `ALV`
  - `MIH792896703`: same values
- **No AMU with a complete 103 yet.** The value of `kzbAV` for U-Kasse is unknown.

### Observations from AMU PDFs

- On the standard TB: `Routing: bAV`, `Durchführungsweg: Unterstützungskasse`, `Finanzierungsart: AG`, and VN = "Überbetriebliche Unterstützungskasse AXA e.V." (with a separate "Trägerunternehmen").
- Some TBs use the U-Kasse layout **without a Data Matrix**.
- One AMU document had an **email cover page**: "Rückdeckungsversicherung (unmittelbare Pensionszusage)". So an email page does **not** mean E00.
- U-Kasse is one of the five bAV Durchführungswege. AMU is a subset of bAV.

---

## 9. Proposed design

### Principle

AMB, AMU, AMD and AME need **no new training**. These SSTs are a property of the **document**, not a page class. If AMB became a page class, the Ausweis page of an AMB document would also be labelled AMB. The current model keeps doing its job (document boundaries + A00, G07, MAD, TB0), and the rest is done with rules.

### Logic

```
Is it bAV?
├── yes → Is it U-Kasse or Rückdeckung?
│         ├── yes → AMU
│         └── no  → AMB
└── no  → TB check
          ├── broken + DÄV → AMD
          ├── broken       → AME
          └── intact       → A00
```

### Signal sources (in priority order)

| Priority | Source | When it works | Signal |
|---|---|---|---|
| 1 | Payload of 103 | Complete TB | `kzbAV` (main), `kzFinanz` and `c_schicht` (confirmation), `mandantId` (DÄV) |
| 2 | `AXA01` code (form number) | Always, even without a TB | Table "form number → SST" from labelled data **(to be built)** |
| 3 | Regex on OCR of the TB text page | Incomplete TB, text pages only, or U-Kasse layout | `Routing: bAV`, `Durchführungsweg:`, `Finanzierungsart:`, `Gesellschaft:`, product code `-<schicht>-` |
| 4 | LLM | Free text only (emails, letters) | |
| 5 | Manual (Bestand) | | |

- **Regex instead of an LLM** for the TB, because the text is machine-generated with a fixed layout, and regex is deterministic, free and keeps data in-house. Before matching: lowercase, `ü→u`, collapse whitespace. Always check several independent signals together.
- **`c_schicht` alone is not enough.** Riester is also Schicht 2 but is not bAV.
- When the TB is broken and 103 cannot be read, DÄV comes from OCR of the text page ("Gesellschaft: DÄV", "Deutsche Ärzteversicherung AG") or from the letterhead of the Angebot pages.
- Rules must run on **all pages of the document**, not only on TB0. The model has never seen the U-Kasse TB layout.

### Place in the pipeline

```
Render (shared)
  → Prepare: separate batch_id, empty page_class, no assign_splits
  → OCR + CLIP cache
  → Stage 1 + 2 (predict only, no retraining)
  → Signals: TB Data Matrix | System lookup (AOR, A01) | Quality, e-mail (LNS, E00)
  → Rule engine → SST per document
```

Changes needed for a prediction-only batch:
- `data.py`: accept an empty `page_class` (`has_label=False`, `class_target=-100`).
- Skip `assign_splits`.
- Skip the population and features hash checks. Keep the stage 1 checkpoint sha check.

The SST label from Snowflake is used **for evaluation only**.

---

## 10. Scripts

### `classification/analysis/tb_datamatrix_check.py`

```
pixi run -e dev python -m life_docai_analytics_antrag.classification.analysis.tb_datamatrix_check --batch-id antrag_life_A00_G07_MAD --limit-docs 300
```

- Paths come from `PipelineConfig`. Output goes to `classification/results/<batch_id>/tb_check/`.
- `--predictions`: production view (predicted boundaries). Without it, pages are grouped by `doc_id`.
- `--save-payloads`: write payloads to `tb_check/payloads/`.
- The fallback runs only for documents with missing chunks: zxing on contrast-stretched, sharpened and 2x upscaled images.
- Columns: `page_datamatrix`, `containers`, `complete`, `missing_chunks`, `undecodable`, `conflicting_chunks`, `fallback`, `broken_before_fallback`, `broken`, `derived_class`, `other_datamatrix`, `match`
- At the end, a container table is printed; unknown containers are marked `<- unknown`.

### `classification/analysis/test/tb_datamatrix_check_test.py` (for AMB and AMU)

Known issues:
1. The fallback runs on **every** page, because the condition is evaluated per page. This is slow, especially with 2x upscaling at 300 DPI.
2. The page-level columns (`broken`, `fallback_used`) are meaningless. Only the `mid_...` columns are valid.
3. Page payloads go to `payloads/AMB|AMU/...` (single-page containers only). Document payloads go to `payloads/<MID>/`, **without** the group name.
4. A MID is a stack, not a document. If a stack contains two TBs, their chunks get mixed (`mid_conflicting_chunks`).
5. The MID is found with a regex on the path, which is fragile. A direct join on `image_path` is safer.
6. Drops and errors are silent (`except Exception`, empty or ambiguous MID).
7. `raw_codes` writes the full Base64. The CSV gets large and contains personal data.

---

## 11. Open questions

1. **AMU:** what value does `kzbAV` have for U-Kasse? We need an AMU with a complete 103. If this field does not separate AMB from AMU, check the VN in the Partner container (100) ("Unterstützungskasse e.V.").
2. **Nina:** Direktzusage with Rückdeckungsversicherung → AMU or AMB? Where does Riester go? Definition of AMW? For AOR, a duplicate of what?
3. **Martin:** why are AMB and AMU always Bestand?
4. **Colleague:** what is container `0000010110`?
5. Is the input channel (email vs. scan) available in the metadata?
6. Cause of `20375`: scan quality on pages 3–5?

---

## 12. Next steps

1. ~~Values of `c_schicht` and `kzFinanz` in A00~~ **done** (section 6). For more documents, run the same command on the payloads of a larger batch:
   ```
   grep -hoE "<pcInFilterDef:(kzbAV|kzFinanz|c_schicht)>[^<]*" packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD/tb_check/payloads/*_0000000103.bin | sort | uniq -c
   ```
2. Find an AMU with a complete 103 and check its `kzbAV`. Also find a **Riester** document to check its `c_schicht` and `kzbAV`.
3. Build the table "`AXA01` form number → SST": which SSTs each form appears in (A00, AMB, AMU).
4. Regex on OCR of the TB text page, and measure coverage on AMB/AMU and false positives on A00.
5. Run with `--predictions` to measure the production gap (predicted boundaries vs. `doc_id`).
6. Count how many of the 2260 AMB documents have a complete vs. incomplete TB.

---

## 13. Data protection

The `.bin` files and TB pages contain personal data (date of birth, sex, name, address, sometimes health information). After inspection they should stay in a restricted location or be deleted.

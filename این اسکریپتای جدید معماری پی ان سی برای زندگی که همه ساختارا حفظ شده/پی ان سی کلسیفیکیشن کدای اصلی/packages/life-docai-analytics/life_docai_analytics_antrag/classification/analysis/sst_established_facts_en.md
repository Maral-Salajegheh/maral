# Established facts: Antrag SSTs, Technikblatt and data

> Only facts that were stated or observed in the conversation, each with its source.
> Sources: **Process doc** = Kaputten.pdf (process description for broken Anträge) · **Nina** = business answers from Nina · **Maral** = stated by Maral · **Data** = observed in our runs and files.
> Nothing in this file is a design proposal.

---

## 1. Scope of the current model and data

| Fact | Source |
|---|---|
| The current model was trained on about 413,000 pages with four page classes only: A00, G07, MAD, TB0 | Maral |
| A00, MAD and G07 are much larger in Snowflake than what was used: only a sample of about 12,000 was taken for training; the first 6,000 were an initial check | Maral |
| TB0 pages sit inside A00 documents | Maral |
| The current batch (`antrag_life_A00_G07_MAD`) contains only clean A00, G07 and MAD documents. No other SST is in it | Maral |
| The other SSTs were only discovered recently and are not yet in the pipeline | Maral |
| The SST is a **document** label, not a stack label | Maral |
| The real page order is `image_id` (checked in ANDB). Documents in a stack can be interleaved | Maral |

## 2. SST volumes (category Antrag, from the Snowflake table)

37 SSTs in category Antrag (query `Investigate SST.sql`, 23.11.2026). The 26 rows visible on screen:

| SST | Documents | Stacks |
|---|---|---|
| A00 | 99,734 | 49,594 |
| MAD | 35,035 | 34,892 |
| G07 | 34,459 | 34,407 |
| LNS | 21,447 | 21,433 |
| AME | 6,075 | 6,019 |
| AOR | 5,689 | 5,624 |
| E00 | 2,725 | 2,721 |
| AMB | 2,260 | 2,198 |
| AMD | 1,322 | 1,307 |
| A01 | 1,205 | 1,203 |
| RIM | 496 | 494 |
| A06 | 266 | 266 |
| AMU | 110 | 108 |
| AMW | 47 | 47 |
| K07 | 47 | 45 |
| E04 | 27 | 27 |
| V04 | 15 | 15 |
| FIN | 13 | 13 |
| H02 | 9 | 9 |
| A07 | 8 | 8 |
| E01 | 8 | 8 |
| U03 | 7 | 7 |
| A17 | 5 | 5 |
| I00 | 5 | 4 |
| S02 | 4 | 4 |
| N00 | 2 | 2 |

- The remaining 11 SSTs are not visible in the screenshot.
- Earlier counts in this conversation (A00 ~35,035, LNS ~6,075, AME ~5,689, AOR ~2,725) were **misread by one row**. The table above is the corrected reading.
- A00: about 2 documents per stack (consistent with A00 + Technikblatt having two `doc_id`s).
- AMB: 2,260 documents in 2,198 stacks, AMU: 110 in 108. So about 62 AMB stacks and 2 AMU stacks contain two documents of the same SST. They are never shared with another SST (checked by Maral).

---

## 3. Process facts (Process doc)

### Classes used by the clerk

| Class | Meaning |
|---|---|
| **21** | TB: the clerk puts the TB pages together and sends them to the Analyse system |
| **40** (`Bestand/Default`) | TB could not be repaired, or the clerk **staples all pages together**; the document is then processed manually |

### Per SST

| SST | What happens | How the clerk recognises it |
|---|---|---|
| **AME** | TB broken or partial → class 21 → AMA check that the Antrag is not already registered → TB sent to Analyse. If it comes back as class 40 (not repaired): all pages stapled, class 40, Antrag registered **manually** with SST AME. For an AXA or DBV logo: name and Orga-Nr. from the ID, comment *"TB kaputt trotz Analyse"* | Result of the Analyse system |
| **AOR** | Search in AMA (name + date of birth) and in PePe / Life Factory (VSNR). Compare the new Antrag with the one in the file: barcode, date at the bottom right of the form, amounts. If **identical**: staple, class 40, **AOR + VSNR**. If a barcode (or a reference to an earlier Antrag) is present, this check must be done. Snowflake description of AOR: *"Original-Antrag zu EES/Fax"* | Comparison with the file in the system |
| **E00** | Broker reminder email (example: *"wir möchten Sie höflich an beigefügten Vorgang erinnern … bereits am … zugestellt"*). Processed **like AOR**, but registered as **E00** + VSNR | Email subject or body contains a reminder |
| **AMD** | Antrag from Deutsche Ärzteversicherung. TB checked: class 21 if complete or partial, class 40 if still broken after Analyse. Registered manually as AMD because the DÄV Antrag is not in the system. Comments: *"TB kaputt"*, *"TB kaputt trotz Analyse"*, *"TB nicht vorhanden"* | **DÄV logo top left** on the Antrag page |
| **A01** | Supplementary documents for an earlier Antrag (signature, doctor's report, ID, Technikblatt). VSNR registered if present in the email | Email usually says *"ergänzende Angaben"* |
| **AMB** | bAV, §63 EStG, Direktversicherung. **Direktversicherung has priority over AOR and barcode** → always AMB | Hint in the email or in the Technikblatt |
| **AMU** | Unterstützungskasse | Hint in the email or in the Technikblatt |
| **AMW** | Rahmenvertrag (e.g. BW) | Hint in the email or in the Technikblatt |

---

## 4. Business confirmations (Nina)

| Confirmed rule | Note |
|---|---|
| Normal Antrag + TB broken → **AME** | |
| DÄV + TB broken → **AMD** | DÄV with an intact TB is a normal **A00** |
| bAV → **AMB**, **always**, TB intact or broken | |
| U-Kasse → **AMU**, **always**, TB intact or broken | |
| Rahmenvertrag → AMW | **Struck out by Nina.** When AMW applies is still open |
| Duplicate (Dublette) → **AOR** | |
| Reminder email → **E00** | |
| Supplementary documents for an earlier Antrag → **A01** | |

---

## 5. Facts stated by Maral

| Fact |
|---|
| **LNS** = "neu scannen" (must be rescanned) |
| In ANDB, `sfdoc_class` / `page_class` of these SSTs differ from A00 / TB0; they are mostly `Bestand/Default` |
| These SSTs are **not segmented**: even if a document contains G07, MAD or TB0 pages, all its pages share **one `doc_id`** |
| A MID has two `doc_id`s only for A00 documents that contain a Technikblatt |
| A stack can contain several `doc_id`s |
| **AMB and AMU stacks are never shared with another SST** (checked by Maral). An AMB stack contains only AMB documents, an AMU stack only AMU documents |
| Each AMB / AMU document has a single `doc_id`, even with G07, MAD or TB0 pages inside. Most such stacks contain one document; some contain two documents of the same SST (about 62 AMB stacks, about 2 AMU stacks, see section 2) |
| **Mixed stacks exist**, e.g. a stack with A00 + MAD + AME: several `doc_id`s, one of them AME |
| **E00** = only the reminder email is the difference; the rest is A00 content, and **all its pages are `Bestand`** in Snowflake |
| `Bestand/Default` (class 40) does not by itself mean the TB is broken (AMB/AMU are always Bestand) |

---

## 6. Observed examples

### Mixed stack: one email, four Technikblätter

| Page | Content | SST |
|---|---|---|
| 1 | Email: *"anbei erhalten Sie die fehlenden Technikblätter für folgende Anträge"* (four customers) | AOR (together with page 2) |
| 2 | TB, customer 1, earlier Antragsdatum | **AOR** |
| 3 | TB, customer 2 | **AME** |
| 4 | TB, customer 3 | **AME** |
| 5 | TB, customer 4 (DBV) | **AME** |

- Pages 2–5 have the same content type (TB); the SST difference comes only from the lookup in the system.
- The email signature contained *"Fachteam LV/bAV"* and *"bAV Antragsservice"*, but none of the documents was AMB. **"bAV" in an email signature is not a bAV signal.**
- A late-sent TB was labelled AOR here, although the A01 rule ("supplementary documents incl. Technikblatt") would also fit. **Open question.**

### AMB / AMU documents (PDFs)

- AMB TB: products `ALVSBV-2-2025` (BU), `ALVRX-2-2025` (Relax Rente), **Schicht 2**.
- AMU TB (standard layout): `Routing: bAV`, `Durchführungsweg: Unterstützungskasse`, `Finanzierungsart: AG`, VN = *"Überbetriebliche Unterstützungskasse AXA e.V."*, plus a separate *Trägerunternehmen*.
- Some AMU Technikblätter use a **different layout without any Data Matrix** (coloured bars, heading "5. Technikblatt" / "Technikblatt").
- One AMU document had an **email cover page** (*"Rückdeckungsversicherung (unmittelbare Pensionszusage)"*). An email page therefore does not mean E00.
- U-Kasse is one of the five bAV Durchführungswege.

---

## 7. Technikblatt and Data Matrix (Data, colleague code `AXATechnikblatt.spl`)

### Data Matrix types

| Code | Content |
|---|---|
| `AXA03` | Data: `AXA03` + container (10) + chunk no. (3) + total (3) + Base64. Chunks concatenated in order → Base64 → gzip |
| `AXA01` | Form identifier: form number + (version) + `P<form page>`, e.g. `AXA01D21006844_(01.23)P03`. On **all AXA forms**, not only the TB. Contains **no** Antrag number or barcode |
| `NICHTGWGIVRELEVANT` | GwG flag, repeated on every TB page when present |

### Containers

| Container | Meaning | Required |
|---|---|---|
| `0000000100` | Partner | optional |
| `0000000101` | Zusatz | optional |
| `0000000102` | Antrag | required |
| `0000000103` | Angebot | required |
| `0000010110` | unknown; seen in 24 of 300 A00 documents, no effect on the decision | — |

AXA decision (`tbCheckDocClassForContainer`): 102 or 103 incomplete → Bestand (*"Datamatrix fehlt / defekt, bitte als Bestand (AME,AMD,AMB,...) weitergeben"*); Zusatz missing → *Antrag/Technikblatt Partner und Antragsdaten*; Partner missing → *Antrag/Technikblatt Partnerdaten*; all present → *Antrag/Technikblatt*.

### Page structure of the standard TB

| Page | Data Matrix codes |
|---|---|
| Text pages (usually 2, sometimes 3) | 1 `AXA01` (+ 1 GwG if flagged); **no `AXA03` data** |
| Grid pages | 1 `AXA01` + up to 12 `AXA03` = 13 (14 with GwG) |
| Last grid page | 1 `AXA01` + remaining chunks |

- TB form numbers: `D21011325`, `D21011326`, `D21012001`.
- Text page 1 **prints**: Produkt, Schicht, Gesellschaft, Routing, Durchführungsweg, Finanzierungsart, Zugangskanal, Zugangsweg, VN data, Barcode.
- Some documents contain only the TB text pages (no grid pages) → no payload possible.

### Payload content

- Container 103: XML `soProposalCalculateCustomerRequest` (FJA), tags `pcInFilterDef:*`.
- Container 102: XML `soProcessApplicationImportRequest`, tags `applicationFilterDef:*` and `lifeContract:*` (`companyId` = `ALV`); no bAV-related tags found.
- Product code pattern: the middle digit is the Schicht (`DLVSBV-3-2025`, `ALVSBV-2-2025`).

| Field (103) | A00 sample (201 files) | AMB (`MIH735396335`, `MIH792896703`) |
|---|---|---|
| `kzbAV` | **never present** | `DIREKT` |
| `c_schicht` | `SCHICHT3` (195), `SCHICHT1` (6), no `SCHICHT2` | `SCHICHT2` |
| `kzFinanz` | `NICHTRELEVANT` (115), absent in 86 | `AGANFINANZIERT` |
| `mandantId` | `ALV` / `DAEV` | `ALV` |

- `MIH735396335` (AMB) has a **complete, intact TB** and is still AMB (confirms Nina's "always").
- No AMU with a complete 103 has been seen yet.
- No Riester document has been seen yet.

---

## 8. Decoder validation results (Data)

| Run | Grouping | Result |
|---|---|---|
| 1 | `segmentation_target` | 300 documents, 97 % match, 9 mismatches (2 mapping errors, 5 decode failures, 1 interleaved TB `37388`, 1 harmless) |
| 2 | mapping `doc_id` + fallback | **297 / 300 correct**; 3 mismatches: 2 mapping errors (`mkkvb0300000297879`, `mklia0000000027334`), 1 decode failure (`mklia0000000020375`) |

- Containers in run 2: 102 and 103 in 298 documents, 101 in 186, 100 in 185.
- `libdmtx` is not available on the server (no `apt-get`; not on conda-forge).
- AMB/AMU test (separate render, 300 DPI): 734 images, 115 with a Data Matrix; most AMB documents have no grid pages; 4 MIDs with complete containers, 2 with a complete 103 (both AMB).

---

## 9. Open questions

1. When does AMW apply (Rahmenvertrag rule struck out)?
2. A late-sent Technikblatt: A01 or AOR?
3. Priority when several rules apply (e.g. bAV + duplicate: Process doc says AMB wins).
4. What value does `kzbAV` take for U-Kasse? Need an AMU with a complete 103.
5. How is Riester recorded (`c_schicht`, `kzbAV`)?
6. Apart from AMB and AMU, which SSTs appear in mixed stacks, and which always form a whole stack?
7. Where is the result of the Analyse system (class 21 / 40) stored?
8. Is there automated access to AMA / PePe for the AOR check?
9. What is container `0000010110`?
10. Is the input channel (email vs. scan) available in the metadata?

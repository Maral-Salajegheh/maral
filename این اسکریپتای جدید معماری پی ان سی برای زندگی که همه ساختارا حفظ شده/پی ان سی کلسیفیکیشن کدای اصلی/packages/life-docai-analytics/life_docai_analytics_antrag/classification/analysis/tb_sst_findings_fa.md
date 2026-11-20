# Technikblatt (TB) و SST های جدید: یافته‌ها تا ۲۰ نوامبر ۲۰۲۶

> این فایل فقط چیزهایی را دارد که در داده دیده یا اجرا شده‌اند. حدس‌ها با **(فرض)** علامت خورده‌اند.

---

## ۱. هدف

وقتی یک سند ِ جدید وارد می‌شود، pipeline باید SST آن را تشخیص بدهد. داده‌ی آموزشی ِ فعلی فقط A00، G07 و MAD را دارد (TB0 داخل A00 است). SST های دیگر (AME، AMD، AMB، AMU، AMW، AOR، E00، A01، LNS) با A00 بیشتر در **فرایند** فرق دارند، نه در **محتوا**.

- SST برچسب ِ **سند** است، نه stack.
- در ANDB، `sfdoc_class` ِ این SST ها بیشتر `Bestand/Default` (کلاس 40) است.

---

## ۲. قانون‌های SST (تأییدشده توسط Nina)

| شرط | SST |
|---|---|
| عادی + TB خراب | AME |
| DÄV + TB خراب | AMD |
| bAV (همیشه، چه TB سالم باشد چه خراب) | AMB |
| U-Kasse (همیشه، چه TB سالم باشد چه خراب) | AMU |
| Rahmenvertrag | AMW، **خط خورده، هنوز باز** |
| تکراری | AOR |
| ایمیل یادآوری | E00 |
| مدرک تکمیلی | A01 |
| باید دوباره اسکن شود | LNS |

**تأیید در داده:** `MIH735396335` (AMB) TB ِ کامل و سالم دارد (`mid_broken=False`) و با این حال AMB است. پس «TB سالم» به‌تنهایی یعنی A00 **نیست**. اول باید bAV را چک کرد.

---

## ۳. ساختار ِ Technikblatt

### دو قالب ِ مختلف

1. **TB ِ استاندارد ِ AXA**: سربرگ «!! Technikblatt – zwingend mit dem Antrag einreichen !!»، با Data Matrix.
2. **TB ِ ابزار ِ پیشنهاد ِ bAV یا U-Kasse**: نوارهای رنگی، تیتر «5. Technikblatt» یا «Technikblatt»، و عنوان‌هایی مثل «Angestelltenversorgung als arbeitgeberfinanzierte Unterstützungskasse». این قالب **هیچ Data Matrix ای ندارد**. در AMU دیده شد.

### صفحه‌های TB ِ استاندارد

| صفحه | Data Matrix |
|---|---|
| صفحه‌های متنی (معمولاً ۲، گاهی ۳) | ۱ کد `AXA01...` (+ اگر GwG flag باشد، ۱ کد `NICHTGWGIVRELEVANT`) |
| صفحه‌های grid | ۱ کد `AXA01...` + تا ۱۲ کد `AXA03` = **۱۳** (با GwG، **۱۴**) |
| صفحه‌ی آخر grid | ۱ کد `AXA01...` + chunk های باقی‌مانده |

نمونه‌ها از سندهایی با `match=True`:

```
1 1 13 13 10          ← 2 صفحه‌ی متنی، 3 صفحه‌ی grid
2 2 14 14 7           ← با GwG flag
2 2 2 14 14 14 9      ← 3 صفحه‌ی متنی + GwG
1 0 1 13 13 13 8      ← کد ِ یک صفحه خوانده نشده، ولی match=True
```

- صفحه‌های متنی **هیچ داده‌ی AXA03 ای ندارند**. همه‌ی container ها روی صفحه‌های grid هستند.
- صفحه‌ی ۱ ِ متنی اطلاعات ِ اصلی را **چاپ** می‌کند: `Produkt:`، `Schicht`، `Gesellschaft:`، `Routing:`، `Durchführungsweg:`، `Finanzierungsart:`، `Zugangskanal:`، `Zugangsweg:`، `VN Nachname:` و غیره.
- بعضی سندها **فقط صفحه‌های متنی ِ TB** را دارند و صفحه‌ی grid ندارند. این‌ها هیچ payload ای نمی‌دهند.

---

## ۴. فرمت ِ Data Matrix ها

### `AXA03` (داده)

```
AXA03 | container (10) | شماره‌ی chunk (3) | تعداد کل (3) | Base64 ...
```

chunk ها به ترتیب شماره کنار هم گذاشته می‌شوند، بعد Base64 decode و بعد gzip باز می‌شود.

| container | معنی | لازم؟ |
|---|---|---|
| `0000000100` | Partner | اختیاری |
| `0000000101` | Zusatz | اختیاری |
| `0000000102` | Antrag | **لازم** |
| `0000000103` | Angebot | **لازم** |
| `0000010110` | ناشناخته | در ۲۴ سند از ۳۰۰ دیده شد، روی تصمیم اثری ندارد |

### `AXA01` (شناسه‌ی فرم)

```
AXA01 + شماره‌ی فرم + (نسخه) + P<شماره‌ی صفحه‌ی فرم>
مثال: AXA01D21006844_(01.23)P03
```

- روی **همه‌ی فرم‌های AXA** هست، نه فقط TB. همان Data Matrix ِ کوچک ِ پایین یا بالای صفحه است.
- فرم‌های TB: `D21011325`، `D21011326`، `D21012001`
- فرم‌های دیگری که در AMB دیده شدند: `D21006836`، `D21012882`، `D21006947`، `D21006844`، `D21012001`
- **شماره‌ی Antrag یا barcode را ندارد.** پس نمی‌شود در پروداکشن صفحه‌های یک TB را با آن گروه‌بندی کرد.
- ولی **نوع ِ فرم** را می‌دهد، و این یک سیگنال ِ قطعی ِ بالقوه برای bAV است (بخش ۹).

### `NICHTGWGIVRELEVANT`

یک flag ِ GwG است که در بعضی سندها روی همه‌ی صفحه‌های TB تکرار می‌شود.

---

## ۵. تصمیم ِ AXA از روی container ها (`tbCheckDocClassForContainer`)

```
102 یا 103 ناقص        → Bestand   («Datamatrix fehlt / defekt, bitte als Bestand (AME,AMD,AMB,...) weitergeben»)
Zusatz (101) نیست      → Antrag/Technikblatt Partner und Antragsdaten
Partner (100) نیست     → Antrag/Technikblatt Partnerdaten
همه هستند              → Antrag/Technikblatt
```

---

## ۶. محتوای payload ها

### container ِ 103 (Angebot)

- XML: `soProposalCalculateCustomerRequest` (namespace `xmlns.fja.com`)، تگ‌ها `pcInFilterDef:...`
- فیلدهای کلیدی:

| فیلد | A00 (نمونه‌ی DÄV BU) | AMB (`MIH735396335`، `MIH792896703`) | معنی |
|---|---|---|---|
| `pdId` / `tfId` | `DLVSBV_3_2025` | (Rente، Schicht 2) | کد محصول |
| `c_schicht` | `SCHICHT3` | `SCHICHT2` | لایه‌ی بازنشستگی |
| `kzbAV` | **نیست** | **`DIREKT`** | bAV + Durchführungsweg |
| `kzFinanz` | `NICHTRELEVANT` | `AGANFINANZIERT` | سهم ِ کارفرما |
| `mandantId` | `DAEV` | `ALV` | AXA Leben یا DÄV |
| `c_gesellschaft` | `DAEV` | `AXA` | |
| `c_zugangsweg` | `AXA_ZUGANGSWEG_1044` | `AXA_ZUGANGSWEG_2000` | |
| `vertriebsweg` | `AXA_KEINE_ANGABE` | `AXA_KEINE_ANGABE` | |

- بیشتر ِ فرق‌های دیگر در `diff` به **نوع ِ محصول** مربوط‌اند (BU در مقابل Rente)، نه به bAV. مثلاً `berufsgruppe` و `risikogruppe` فقط در BU هستند، و `fondsId`، `garantieniveau` و `kzRente` فقط در Rente.
- الگوی کد ِ محصول: **عدد ِ وسط همان Schicht است**: `DLVSBV-3-2025`، `ALVFRV-3-2025` (A00)، `ALVSBV-2-2025`، `ALVRX-2-2025` (AMB و AMU).

### container ِ 102 (Antrag)

- XML: `soProcessApplicationImportRequest`، تگ‌ها `applicationFilterDef:...` و `lifeContract:...`
- `lifeContract:companyId` = `ALV`
- در جستجوی ِ کلمه‌های bAV، Durchführungsweg، Routing و Finanz **هیچ نتیجه‌ای** نداشت. پس برای bAV، **103 منبع ِ اصلی است**.

### نمونه‌ی A00 (201 فایل ِ 103، اجرای ِ قدیمی با `--payload-dir`)

| فیلد | A00 (201 فایل) | AMB (2 سند) |
|---|---|---|
| `kzbAV` | **هیچ‌وقت** | `DIREKT` |
| `c_schicht` | `SCHICHT3` (195)، `SCHICHT1` (6)، **هیچ `SCHICHT2`** | `SCHICHT2` |
| `kzFinanz` | `NICHTRELEVANT` (115)، در 86 فایل نیست | `AGANFINANZIERT` |

- در این نمونه، هر سه فیلد A00 را از AMB جدا می‌کنند.
- **`kzbAV` سیگنال ِ اصلی است.** در A00 هرگز نیامده، و از نظر مفهومی هم مستقیماً «bAV» است.
- **`kzFinanz` فقط وقتی معنی دارد که وجود داشته باشد.** در 86 فایل نیست، احتمالاً بسته به نوع محصول (مثلاً BU). پس «نبودن ِ `kzFinanz`» یعنی bAV نیست **نیست**.
- **`SCHICHT2` به‌تنهایی کافی نیست.** در این نمونه هیچ Riester ای نبوده، پس معلوم نیست Riester چطور ثبت می‌شود. از `c_schicht` فقط به‌عنوان تأیید استفاده شود.
- اندازه‌ی نمونه کوچک است: حدود ۲۰۰ A00 و ۲ AMB.
- نکته‌ی فنی: اگر `<` اول ِ الگوی `grep` نباشد، تگ‌های بسته هم شمرده می‌شوند و خط‌هایی بدون مقدار در خروجی می‌آیند.

### قانون ِ فعلی برای TB ِ کامل

```
kzbAV پر است    → bAV  (AMB؛ برای AMU هنوز تأیید نشده)
kzbAV نیست      → bAV نیست → TB check (AME / AMD / A00)
تأیید:          c_schicht = SCHICHT2  و  kzFinanz = AGANFINANZIERT
```

---

## ۷. اعتبارسنجی ِ decoder روی batch ِ A00/G07/MAD

### اجرای ۱ (گروه‌بندی با `segmentation_target`)

- ۳۰۰ سند، match ۹۷٪، ۹ mismatch:
  - خطای mapping (همه‌ی صفحه‌ها بدون Data Matrix): `36045`، `mkkvb…297879`
  - decode ِ ناموفق: `20375`، `50078`، `27665`، `16770`، `55048`
  - TB ِ تکه‌تکه به‌خاطر ترتیب ِ درهم ِ `image_id`: `37388`
  - ما بیشتر از AXA خواندیم (بی‌ضرر): `48136`

### اجرای ۲ (گروه‌بندی با `doc_id` + fallback)

- **297 از 300 درست**، ۳ mismatch:
  - خطای mapping: `mkkvb0300000297879`، `mklia0000000027334`
  - decode ِ ناموفق: `mklia0000000020375` (`0 0 12 12 7`، `missing 102:1`، Partner اصلاً دیده نشده)
- container ها: 102 و 103 در 298 سند، 101 در 186، 100 در 185، `0000010110` در 24.
- به‌خاطر تغییر ِ گروه‌بندی، نمونه‌ی ۳۰۰ تایی با اجرای ۱ یکی نیست.

### تحلیل ِ `20375`

- `0 0` روی صفحه‌ی ۱ و ۲ بی‌ضرر است، چون آنجا فقط `AXA01` هست.
- مشکل ِ اصلی در صفحه‌های grid است: روی هر کدام یک کد کم است، Partner کلاً نیست، و یک chunk از 102 هم کم است.
- **(فرض)** کیفیت ِ اسکن. باید صفحه‌ی ۳ تا ۵ را دید.

### درس‌ها

- ترتیب ِ واقعی ِ صفحه‌ها `image_id` است (در ANDB چک شد). سندها ممکن است درهم باشند (مثل `37388`). برای اعتبارسنجی، `doc_id` ِ mapping درست است. در پروداکشن `doc_id` نیست.
- `libdmtx` روی سرور نیست (`apt-get` وجود ندارد، conda-forge هم `libdmtx` ندارد). از fallback حذف شد.
- `zlib.error` در `except` نبود و برنامه را متوقف می‌کرد. اضافه شد. ستون `conflicting_chunks` هم اضافه شد (یک شماره‌ی chunk با دو متن ِ مختلف، یعنی دو TB قاطی شده‌اند).

---

## ۸. یافته‌های AMB و AMU (رندر ِ جدا، ۳۰۰ DPI)

- 734 تصویر، 115 تصویر با Data Matrix.
- MID درست پیدا شد (mapping مشکلی نداشت).
- بیشتر ِ سندهای AMB **صفحه‌ی grid ندارند** (`axa03_chunk_count = 0`)، پس payload ندارند.
- 4 MID container ِ کامل داشتند. دو تا 103 ِ کامل دارند و **هر دو AMB هستند**:
  - `MIH735396335`: `kzbAV=DIREKT`، `SCHICHT2`، `AGANFINANZIERT`، `ALV`
  - `MIH792896703`: همان مقدارها
- **هنوز هیچ AMU ای با 103 ِ کامل نداریم.** پس مقدار ِ `kzbAV` برای U-Kasse معلوم نیست.

### مشاهده‌ها از PDF های AMU

- روی TB ِ استاندارد: `Routing: bAV`، `Durchführungsweg: Unterstützungskasse`، `Finanzierungsart: AG`، و VN = «Überbetriebliche Unterstützungskasse AXA e.V.» (با یک «Trägerunternehmen» ِ جدا).
- بخشی از TB ها قالب ِ U-Kasse **بدون Data Matrix** دارند.
- یک سند ِ AMU **جلد ِ ایمیل** داشت: «Rückdeckungsversicherung (unmittelbare Pensionszusage)». پس صفحه‌ی ایمیل به معنی E00 **نیست**.
- U-Kasse یکی از پنج Durchführungsweg ِ bAV است. AMU زیرمجموعه‌ی bAV است.

---

## ۹. طراحی ِ پیشنهادی

### اصل

برای AMB، AMU، AMD و AME، **training ِ جدید لازم نیست**. این SST ها ویژگی ِ **سند** هستند، نه کلاس ِ صفحه. اگر AMB را کلاس ِ صفحه کنیم، صفحه‌ی Ausweis ِ یک سند AMB هم برچسب AMB می‌گیرد. مدل ِ فعلی همان کار ِ خودش را انجام می‌دهد (مرز ِ سندها + A00، G07، MAD و TB0) و بقیه با قانون انجام می‌شود.

### منطق

```
آیا bAV است؟
├── بله → آیا U-Kasse یا Rückdeckung است؟
│         ├── بله → AMU
│         └── نه  → AMB
└── نه  → TB check
          ├── خراب + DÄV → AMD
          ├── خراب       → AME
          └── سالم       → A00
```

### منبع ِ سیگنال‌ها (به ترتیب ِ اولویت)

| اولویت | منبع | کی کار می‌کند | سیگنال |
|---|---|---|---|
| ۱ | payload ِ 103 | TB ِ کامل | `kzbAV` (اصلی)، `kzFinanz` و `c_schicht` (تأیید)، `mandantId` (DÄV) |
| ۲ | کد ِ `AXA01` (شماره‌ی فرم) | همیشه، حتی بدون TB | جدول ِ «شماره‌ی فرم → SST» از داده‌ی برچسب‌دار **(باید ساخته شود)** |
| ۳ | regex روی OCR ِ صفحه‌ی متنی ِ TB | TB ِ ناقص، فقط دو صفحه‌ی اول، یا قالب ِ U-Kasse | `Routing: bAV`، `Durchführungsweg:`، `Finanzierungsart:`، `Gesellschaft:`، کد ِ محصول `-<schicht>-` |
| ۴ | LLM | فقط متن ِ آزاد (ایمیل، نامه) | |
| ۵ | دستی (Bestand) | | |

- **regex به‌جای LLM** برای TB، چون متن را ماشین تولید کرده، قالبش ثابت است، و regex قطعی و رایگان است و داده بیرون نمی‌رود. قبل از regex: حروف کوچک، `ü→u`، یکی کردن ِ فاصله‌ها. همیشه چند سیگنال ِ مستقل با هم چک شوند.
- **`c_schicht` به‌تنهایی کافی نیست.** Riester هم Schicht 2 است، ولی bAV نیست.
- وقتی TB خراب است و 103 خوانده نمی‌شود، DÄV از این‌ها معلوم می‌شود: OCR ِ صفحه‌ی متنی («Gesellschaft: DÄV»، «Deutsche Ärzteversicherung AG») یا سربرگ ِ صفحه‌های Angebot.
- قانون‌ها باید روی **همه‌ی صفحه‌های سند** اجرا شوند، نه فقط روی TB0. مدل، TB ِ قالب ِ U-Kasse را ندیده است.

### محل در pipeline

```
Render (مشترک)
  → Prepare: batch_id ِ جدا، page_class ِ خالی، بدون assign_splits
  → OCR + CLIP cache
  → Stage 1 + 2 (فقط predict، بدون training ِ دوباره)
  → سیگنال‌ها: TB Data Matrix | System lookup (AOR، A01) | Quality، e-mail (LNS، E00)
  → موتور ِ قانون → SST برای هر سند
```

تغییرهای لازم برای batch ِ prediction-only:
- `data.py`: `page_class` ِ خالی قبول شود (`has_label=False`، `class_target=-100`).
- `assign_splits` اجرا نشود.
- چک ِ hash ِ population و features رد شود. چک ِ sha ِ checkpoint ِ stage 1 بماند.

برچسب ِ SST از Snowflake **فقط برای ارزیابی** استفاده می‌شود.

---

## ۱۰. اسکریپت‌ها

### `classification/analysis/tb_datamatrix_check.py`

```
pixi run -e dev python -m life_docai_analytics_antrag.classification.analysis.tb_datamatrix_check --batch-id antrag_life_A00_G07_MAD --limit-docs 300
```

- مسیرها از `PipelineConfig` می‌آیند. خروجی در `classification/results/<batch_id>/tb_check/` است.
- `--predictions`: نمای ِ پروداکشن (مرزهای ِ پیش‌بینی‌شده). بدون آن: گروه‌بندی با `doc_id`.
- `--save-payloads`: نوشتن ِ payload ها در `tb_check/payloads/`.
- fallback فقط برای سندهایی که chunk کم دارند: zxing روی تصویر ِ کنتراست‌شده، sharpen‌شده و ۲ برابرشده.
- ستون‌ها: `page_datamatrix`، `containers`، `complete`، `missing_chunks`، `undecodable`، `conflicting_chunks`، `fallback`، `broken_before_fallback`، `broken`، `derived_class`، `other_datamatrix`، `match`
- در آخر جدول ِ container ها چاپ می‌شود و container های ناشناخته علامت `<- unknown` می‌گیرند.

### `classification/analysis/test/tb_datamatrix_check_test.py` (برای AMB و AMU)

ایرادهای شناخته‌شده:
1. fallback روی **همه‌ی** صفحه‌ها اجرا می‌شود، چون شرط در سطح ِ صفحه است. این کار را کند می‌کند، به‌خصوص با ۲ برابر کردن روی ۳۰۰ DPI.
2. ستون‌های ِ سطح ِ صفحه (`broken`، `fallback_used`) بی‌معنی‌اند. فقط ستون‌های `mid_...` معتبرند.
3. payload ِ صفحه‌ها در `payloads/AMB|AMU/...` است (فقط container های تک‌صفحه‌ای). payload ِ سندها در `payloads/<MID>/` است، **بدون** اسم ِ گروه.
4. MID یک stack است، نه یک سند. اگر دو TB در یک stack باشند، chunk هایشان قاطی می‌شوند (`mid_conflicting_chunks`).
5. MID با regex روی مسیر پیدا می‌شود که شکننده است. join ِ مستقیم روی `image_path` مطمئن‌تر است.
6. حذف‌ها و خطاها بی‌صدا هستند (`except Exception`، MID ِ خالی یا ambiguous).
7. `raw_codes` کل ِ Base64 را می‌نویسد. CSV بزرگ می‌شود و داده‌ی شخصی دارد.

---

## ۱۱. سؤال‌های باز

1. **AMU:** `kzbAV` برای U-Kasse چه مقداری دارد؟ به یک AMU با 103 ِ کامل احتیاج داریم. اگر این فیلد AMB و AMU را جدا نکند، باید VN را در container ِ Partner (100) دید («Unterstützungskasse e.V.»).
2. **Nina:** Direktzusage با Rückdeckungsversicherung → AMU یا AMB؟ Riester کجا می‌رود؟ تعریف ِ AMW؟ برای AOR، «تکراری» یعنی تکراری ِ چه چیزی؟
3. **Martin:** چرا AMB و AMU همیشه Bestand هستند؟
4. **همکار:** container ِ `0000010110` چیست؟
5. آیا کانال ِ ورودی (ایمیل یا اسکن) در metadata هست؟
6. علت ِ `20375`: کیفیت ِ اسکن در صفحه‌ی ۳ تا ۵؟

---

## ۱۲. قدم‌های بعدی

1. ~~مقدارهای `c_schicht` و `kzFinanz` در A00~~ **انجام شد** (بخش ۶). برای سندهای بیشتر، همین دستور روی payload های یک batch ِ بزرگ‌تر:
   ```
   grep -hoE "<pcInFilterDef:(kzbAV|kzFinanz|c_schicht)>[^<]*" packages/life-docai-analytics/life_docai_analytics_antrag/classification/results/antrag_life_A00_G07_MAD/tb_check/payloads/*_0000000103.bin | sort | uniq -c
   ```
2. پیدا کردن ِ یک AMU با 103 ِ کامل و دیدن ِ `kzbAV` اش. همچنین پیدا کردن ِ یک **Riester** برای دیدن ِ `c_schicht` و `kzbAV` اش.
3. ساختن ِ جدول ِ «شماره‌ی فرم ِ `AXA01` → SST»: در کدام SST ها هر فرم دیده می‌شود (A00، AMB، AMU).
4. regex ِ صفحه‌ی متنی ِ TB روی OCR، و اندازه‌گیری ِ پوشش روی AMB و AMU و مثبت ِ کاذب روی A00.
5. اجرای `--predictions` برای دیدن ِ فاصله‌ی پروداکشن (مرزهای ِ پیش‌بینی‌شده در مقابل `doc_id`).
6. شمردن ِ اینکه چند تا از 2260 سند AMB، TB ِ کامل یا ناقص دارند.

---

## ۱۳. حفاظت ِ داده

فایل‌های `.bin` و صفحه‌های TB داده‌ی شخصی دارند (تاریخ تولد، جنسیت، نام، آدرس، و گاهی اطلاعات ِ سلامت). بعد از بررسی باید در مسیری با دسترسی ِ محدود بمانند یا پاک شوند.

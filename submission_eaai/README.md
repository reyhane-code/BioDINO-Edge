# بستهٔ کامل Submission — BioDINO-Edge

**ژورنال هدف:** *Engineering Applications of Artificial Intelligence* (EAAI), Elsevier  
**نوع مقاله:** Original Research Article  
**تاریخ آماده‌سازی بسته:** ۲۰۲۶-۱۱-۱۴  
**فرآیند داوری:** Double-anonymized (دوسوکور)

> این پوشه برای **ارسال اولیه (Initial Submission)** به EAAI آماده شده است؛ نه نسخهٔ «پذیرفته‌شده/نهایی» که پس از پذیرش درخواست می‌شود. فایل اصلی manuscript در حالت **single-column** و مطابق تمپلیت رسمی `elsarticle` است.

---

## ۱) خلاصهٔ انطباق با قوانین EAAI

| مورد | قانون رسمی | وضعیت بسته |
|---|---|---|
| قالب manuscript | single-column (شرط desk-reject) | ✅ `preprint,12pt` |
| تعداد صفحات | حداکثر ۵۰ | ✅ **۴۸ صفحه** |
| حجم فایل | کمتر از ۱۰۰ MB | ✅ ~۰.۸۷ MB |
| چکیده | حداکثر ۲۵۰ کلمه | ✅ **۲۴۵ کلمه** |
| تفکیک سهم AI و کاربرد مهندسی در چکیده | الزامی | ✅ جملهٔ صریح در چکیده |
| آکرونیم تعریف‌نشده در title/abstract | ممنوع | ✅ DINOv2/BioCLIP/GPU/CPU/FPS/Fisher-LDA تعریف شده‌اند |
| Keywords | ۱ تا ۶ | ✅ **۶** |
| Highlights | ۳ تا ۵ بولت، هر کدام ≤۸۵ کاراکتر | ✅ **۵ بولت** |
| Declaration of Generative AI | الزامی، قبل از References | ✅ داخل manuscript |
| Funding statement | الزامی | ✅ داخل manuscript + title page |
| CRediT | الزامی | ✅ در title page |
| Declaration of Competing Interest | الزامی | ✅ فایل جداگانه + title page |
| Data Availability (Option C) | الزامی + لینک مخزن | ⏳ فعلاً «upon acceptance»؛ قبل از پذیرش DOI/Zenodo اضافه شود |
| مقاله‌ی بی‌نام | الزامی (double-anonymized) | ✅ `main_anonymized.*` |
| عنوان/نویسندگان جدا | الزامی | ✅ `title_page.*` |
| رفرنس‌ها | بدون سقف تعداد؛ کامل و دارای DOI در صورت وجود | ✅ **۴۱ رفرنس**، بدون تکراری، همه cite شده |

---

## ۲) ساختار پوشه و نقش هر فایل

| فایل | نقش | مرحلهٔ استفاده | کجا در Editorial Manager آپلود شود |
|---|---|---|---|
| `main_anonymized.pdf` | **نسخهٔ بی‌نام manuscript برای داوری** | Initial Submission | فیلد **Manuscript / Article** |
| `main_anonymized.tex` | سورس LaTeX بی‌نام | Initial (در صورت نیاز) و Revision/Production | فیلد **LaTeX source** / `Manuscript` |
| `title_page.pdf` | صفحهٔ عنوان با نام/affiliation/Acknowledgments/CRediT/بیانیه‌ها | Initial Submission | فیلد **Title Page** (جدا از manuscript) |
| `title_page.tex` | سورس صفحهٔ عنوان | Initial / Revision | همراه فایل‌های منبع |
| `highlights.txt` | ۵ بولت Highlights (≤۸۵ کاراکتر) — نسخهٔ متنی | Initial Submission | فیلد **Highlights** |
| `highlights.docx` | همان Highlights به فرمت Word (آمادهٔ آپلود) | Initial Submission | فیلد **Highlights** |
| `cover_letter.md` | Cover Letter (نسخهٔ متنی) | Initial Submission | فیلد **Cover Letter** |
| `cover_letter.docx` | Cover Letter به فرمت Word (آمادهٔ آپلود) | Initial Submission | فیلد **Cover Letter** |
| `declaration_of_competing_interest.md` | بیانیهٔ تعارض منافع (متنی) | Initial Submission | فیلد **Declaration of Interest** |
| `declaration_of_competing_interest.docx` | همان بیانیه به فرمت Word | Initial Submission | فیلد **Declaration of Interest** و/یا ابزار Declarations |
| `data_availability_statement.md` | بیانیهٔ دسترسی به داده/کد (متنی) | Initial + Acceptance | فیلد **Data Availability** |
| `data_availability_statement.docx` | همان بیانیه به فرمت Word | Initial + Acceptance | فیلد **Data Availability** |
| `figures/fig_architecture.pdf` | شکل ۱ (معماری) | Initial (اختیاری) + Production (الزامی) | فیلد **Figure** |
| `figures/fig_reliability.pdf` | شکل ۲ (کالیبراسیون) | همان | فیلد **Figure** |
| `figures/fig_risk_coverage.pdf` | شکل ۳ (risk–coverage) | همان | فیلد **Figure** |
| `figures/fig_tau_sensitivity.pdf` | شکل ۴ (حساسیت τ) | همان | فیلد **Figure** |
| `figures/fig_confusion.pdf` | شکل ۵ (confusion matrix) | همان | فیلد **Figure** |
| `graphical_abstract.pdf` | Graphical Abstract (اختیاری، توصیه‌شده) | Initial Submission | فیلد **Graphical Abstract** |
| `references.bib` | بانک مراجع BibTeX | Revision/Production | همراه سورس LaTeX |
| `elsarticle.cls`, `elsarticle-num.bst` | کلاس و استایل رسمی Elsevier | Production | همراه سورس LaTeX |
| `README.md` | راهنمای داخلی | — | **آپلود نمی‌شود** |

---

## ۳) مرحله به مرحله: چه فایلی، کجا، در چه زمانی

### مرحلهٔ ۱ — پیش از ارسال (Pre-submission)
1. مطمئن شو `main_anonymized.pdf` هیچ نام/ایمیل/affiliation/Acknowledgments/CRediT ندارد.
2. در Properties فایل PDF چک کن فیلد **Author** خالی باشد (فایل بی‌نام کامپایل شده است).
3. `title_page.pdf` را جدا نگه دار؛ این فایل **نباید** با manuscript ادغام شود.
4. `highlights.docx` آماده است؛ در صورت نیاز ویرایش کن (۵ بولت، هر کدام ≤۸۵ کاراکتر).
5. `cover_letter.docx` آماده است؛ فقط تاریخ و در صورت تمایل suggested reviewers را اضافه کن.
6. صفحه‌های شکل‌ها را در `figures/` نگه دار؛ در manuscript برای داوری embed شده‌اند، ولی فایل‌های جداگانه هم لازم است.
7. آدرس پستی کامل affiliation را در `title_page.tex` تکمیل کن (TODO داخل فایل).
8. Inspec classification codes (تا ۶ کد) را برای ورود در سیستم آماده کن.
9. اگر Graphical Abstract می‌دهی، `graphical_abstract.pdf` را بررسی کن (حداقل ۱۳۲۸ پیکسل عرض در حالت رستر؛ PDF برداری مجاز است).

### مرحلهٔ ۲ — Initial Submission در Editorial Manager
آدرس سابمیت: <https://submit.elsevier.com/EAAI>

1. **Article Type:** `Original Research article`
2. **Title / Abstract / Keywords:** از داخل manuscript کپی کن (۶ keyword).
3. **Authors:** نام و affiliation را در سیستم وارد کن (این بخش محرمانه است و در manuscript بی‌نام نمی‌آید).
4. **Upload files** به این ترتیب معمول:
   - `cover_letter.docx` → **Cover Letter**
   - `title_page.pdf` → **Title Page**
   - `main_anonymized.pdf` → **Manuscript**
   - `main_anonymized.tex` (در صورت درخواست سورس) → **LaTeX source**
   - `highlights.docx` → **Highlights**
   - `figures/*.pdf` → **Figure** (هر فایل جداگانه)
   - `graphical_abstract.pdf` → **Graphical Abstract** (اختیاری)
   - `declaration_of_competing_interest.docx` → **Declaration of Interest** (یا از ابزار Declarations استفاده کن)
5. **Declaration of Interest:** ترجیحاً از ابزار رسمی <https://declarations.elsevier.com/> استفاده کن و خروجی DOCX را آپلود کن.
6. **Data Availability:** متن `data_availability_statement.md` را در فیلد مربوطه paste کن.
7. **Suggested Reviewers:** اختیاری؛ در Cover Letter یا فیلد سیستم وارد کن.
8. **Build PDF:** سیستم یک PDF نهایی می‌سازد؛ آن را باز کن و مطمئن شو نام نویسنده در manuscript دیده نمی‌شود.
9. **Submit.**

### مرحلهٔ ۳ — Revision (پس از دریافت نظرات)
1. `main_anonymized.tex` را ویرایش کن (همچنان بی‌نام بماند).
2. `main_anonymized.pdf` را دوباره build کن.
3. فایل **Response to Reviewers** جداگانه بساز و آپلود کن.
4. `title_page.tex/pdf` را در صورت تغییر نویسندگان/affiliation به‌روزرسانی کن.
5. شکل‌های اصلاح‌شده را در `figures/` جایگزین کن.
6. در سیستم، نسخهٔ Revised را به‌عنوان **Manuscript** آپلود کن (نسخهٔ قبلی را overwrite نکن؛ معمولاً سیستم version می‌سازد).

### مرحلهٔ ۴ — Acceptance / Production
1. فایل‌های نهایی سورس را آپلود کن: `main_anonymized.tex`، `references.bib`، `elsarticle.cls`، `elsarticle-num.bst`، `figures/*.pdf`.
2. **Publishing Agreement** را امضا کن.
3. **Proof** را در بازهٔ اعلام‌شده (معمولاً ۲ روز) بررسی و اصلاح کن.
4. DOI آرشیوی (Zenodo) برای کد/مدل بساز و در Data Availability به‌روزرسانی کن.
5. ORCID و اطلاعات نویسنده را تکمیل کن.

---

## ۴) نکات حیاتی آنونیم‌سازی (Double-anonymized)

- فایل manuscript **نباید** شامل نام نویسندگان، ایمیل، affiliation، Acknowledgments یا CRediT باشد. همهٔ این‌ها فقط در `title_page` هستند.
- لینک GitHub در manuscript به `upon acceptance` تغییر داده شده است. اگر لینک ناشناس (anonymous.4open.science) داری، می‌توانی همان را جایگزین کنی.
- در PDF نهایی، فیلد metadata مربوط به Author را خالی نگه دار.
- `README.md`، `cover_letter`، `title_page` و فایل‌های declaration را در فیلد **Manuscript** آپلود نکن.

---

## ۵) چک‌لیست نهایی قبل از کلیک Submit

- [ ] `main_anonymized.pdf` باز شد و هیچ نام/affiliation ندارد.
- [ ] `title_page.pdf` جداگانه آماده است و آدرس پستی کامل دارد.
- [ ] `highlights.txt` ≤ ۸۵ کاراکتر در هر بولت است (۵ بولت).
- [ ] Cover Letter به PDF/DOCX تبدیل شده است.
- [ ] همهٔ ۵ شکل در `figures/` موجود و در manuscript cite شده‌اند.
- [ ] ۴۱ رفرنس، بدون تکراری، همه cite شده.
- [ ] Declaration of Competing Interest آماده است.
- [ ] Data Availability در سیستم وارد شده است.
- [ ] Declaration of Generative AI در manuscript قبل از References موجود است.
- [ ] Article Type = Original Research article.
- [ ] PDF ساخته‌شده در سیستم باز و بازبینی شد.

---

## ۶) کارهای باقی‌مانده (TODO)

1. **آدرس پستی کامل** affiliation در `title_page.tex` (خیابان، ساختمان، کد پستی).
2. **Zenodo DOI** برای کد/مدل‌ها و به‌روزرسانی Data Availability (Option C).
3. **Inspec classification codes** (تا ۶ کد) برای ورود در سیستم.
4. **Declarations tool** (<https://declarations.elsevier.com/>) و آپلود خروجی.
5. **Suggested reviewers** (اختیاری) در Cover Letter.
6. در صورت نیاز سیستم، نسخهٔ PDF از `cover_letter.docx` هم بساز.
7. در صورت تمایل: Graphical Abstract اختصاصی (الان از شکل معماری استفاده شده است).

---

## ۷) منابع رسمی و به‌روز (دسترسی: ۲۰۲۶-۱۱-۱۴)

- EAAI — Guide for Authors:  
  <https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors>
- EAAI — Submit online (Editorial Manager):  
  <https://submit.elsevier.com/EAAI>
- Elsevier — LaTeX instructions and templates:  
  <https://www.elsevier.com/latex>
- Elsevier — Highlights guidelines:  
  <https://www.elsevier.com/researcher/author/tools-and-resources/highlights>
- Elsevier — Graphical abstract guidelines:  
  <https://www.elsevier.com/authors/journal-authors/graphical-abstract>
- Elsevier — Artwork and media instructions:  
  <https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions>
- Elsevier — Generative AI policies for journals:  
  <https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals>
- Elsevier — Research data policy:  
  <https://www.elsevier.com/about/policies-and-standards/research-data>
- Elsevier — Declarations tool:  
  <https://declarations.elsevier.com/>
- CRediT taxonomy:  
  <https://credit.niso.org/>

> نکته: در راهنمای EAAI، دو جمله دربارهٔ قالب وجود دارد: شرط desk-reject می‌گوید **single-column**، و بخش File format می‌گوید double-column فقط برای LaTeX مجاز است. برای بیشترین امنیت، این بسته از **single-column** استفاده می‌کند که هم پیش‌فرض تمپلیت رسمی است و هم با شرط desk-reject هم‌خوان است.

---

## ۸) دستور کامپایل (برای بازتولید PDFها)

از داخل همین پوشه:

```bat
pdflatex -interaction=nonstopmode main_anonymized.tex
bibtex main_anonymized
pdflatex -interaction=nonstopmode main_anonymized.tex
pdflatex -interaction=nonstopmode main_anonymized.tex

pdflatex -interaction=nonstopmode title_page.tex
pdflatex -interaction=nonstopmode title_page.tex
```

خروجی‌ها: `main_anonymized.pdf` و `title_page.pdf`.

---

## ۹) توضیح فایل‌های LaTeX

- `main_anonymized.tex`: نسخهٔ بی‌نام مقاله؛ single-column `preprint,12pt`؛ شامل Related Work، ۵ شکل، ۱۶ جدول، ۴۱ رفرنس؛ Declaration of Generative AI قبل از References.
- `title_page.tex`: صفحهٔ عنوان مستقل (article, 12pt) شامل نویسندگان، affiliation، corresponding author، Acknowledgments، CRediT، Funding، Competing Interest، GenAI declaration.
- `references.bib`: ۴۱ مرجع، بدون تکراری؛ سبک `elsarticle-num` مطابق تمپلیت رسمی number.
- `elsarticle.cls` / `elsarticle-num.bst`: کلاس و استایل رسمی Elsevier (از پکیج دانلودشده).
- `figures/`: پنج شکل برداری PDF برای آپلود جداگانه.
- `graphical_abstract.pdf`: شکل معماری به‌عنوان Graphical Abstract پیشنهادی (اختیاری).

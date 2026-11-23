# راهنمای جامع راه‌اندازی، مهندسی دیتاست و اجرای عملیاتی پایپلاین BioDINO-Edge
### (Comprehensive Execution, Dataset Engineering & Practical Operations Guide)

> **هدف این سند:**  
> این سند راهنمای کامل، همه‌فهم و گام‌به‌گام برای راه‌اندازی و اجرای عملیاتی تمامی کدهای پروژه **BioDINO-Edge** است. چه از لینوکس (اوبونتو/دبیان) استفاده می‌کنید و چه از ویندوز، تمامی دستورات نصب، ساختار کَش هوشمند، مدیریت خط فرمان، اجرای بنچمارک‌ها و سناریوهای رفع اشکال در این مستند تشریح شده است.

---

## ۱. آشنایی با دیتاست‌ها و ساختار پوشه‌ها (Dataset Architecture)

پایپلاین BioDINO-Edge بر روی دو مجموعه داده معتبر بین‌المللی ارزیابی می‌شود:

### الف) دیتاست استاندارد آزمایشگاهی PlantVillage
این دیتاست مرجع استاندارد ارزیابی بینایی ماشین در پاتولوژی گیاهی است. در این پروژه، ۱۱ کلاس حیاتی به عنوان داده‌های کاملاً دیده‌نشده نوظهور ($\mathcal{Y}_{\text{novel}}$) ارزیابی می‌شوند:

```text
research-few-shot/
└── plantvillage/
    └── plantvillage dataset/
        └── color/
            ├── Apple___Apple_scab/             (630 تصویر)
            ├── Apple___Black_rot/              (621 تصویر)
            ├── Apple___Cedar_apple_rust/       (275 تصویر)
            ├── Apple___healthy/                (1,645 تصویر)
            ├── Grape___Black_rot/              (1,180 تصویر)
            ├── Grape___Esca_(Black_Measles)/   (1,383 تصویر)
            ├── Grape___Leaf_blight/            (1,076 تصویر)
            ├── Grape___healthy/                (423 تصویر)
            ├── Potato___Early_blight/          (1,000 تصویر)
            ├── Potato___Late_blight/           (1,000 تصویر)
            ├── Potato___healthy/               (152 تصویر)
            └── ... (۲۷ کلاس پایه دیگر)
```

### ب) دیتاست میدانی مزارع واقعی PlantDoc (In-The-Wild Benchmark)
این دیتاست شامل تصاویر باکیفیت پایین، لرزش دست، نور آفتاب متغیر، خاک و دست کشاورز است که در مزارع هندوستان با دوربین تلفن همراه ثبت شده است.
- **تعداد کلاس‌های مشترک آموزش و آزمون:** ۲۷ کلاس چالش‌برانگیز در پوشه‌های `plantdoc/train` و `plantdoc/test`.

---

## ۲. مکانیزم کَش هوشمند بردارها (`cache_artifacts/`)

یکی از کلیدی‌ترین قابلیت‌های پروژه، **سیستم ذخیره‌سازی آفلاین بردارها (Offline Feature Cache)** است:

* **مسیر فایل کَش:** [`cache_artifacts/pv_novel_embeddings.pt`](../cache_artifacts/pv_novel_embeddings.pt) (حجم: ۲۷.۸ مگابایت).
* **محتوا:** بردار ویژگی‌های استخراج‌شده برای تمام ۹,۳۳۰ تصویر کوئری شامل:
  - ویژگی‌های DINOv2 (۳۸۴ بعد)
  - ویژگی‌های BioCLIP (۵۱۲ بعد)
  - برچسب‌های کلاسی
  - پروتوتایپ‌های متنی استخراج‌شده از پرامپت‌های گیاهی

> [!TIP]
> **صرفه‌جویی زمانی خارق‌العاده:**  
> اگر کَش وجود داشته باشد، آزمون سخت‌گیرانه ۳۰ اپیزودی مونت‌کارلو روی ۹,۳۳۰ تصویر به جای ۴۵ ثانیه پردازش گرافیکی سنگین، در کمتر از **۱ ثانیه** روی CPU اجرا می‌شود!  
> اگر فایل کَش را حذف کنید، کد به صورت خودکار تصاویر خام پوشه `plantvillage` را لود کرده و کَش جدید را می‌سازد.

---

## ۳. مراحل راه‌اندازی گام‌به‌گام محیط پایتون (Environment Setup)

### گام اول: رفتن به دایرکتوری پروژه
در ترمینال سیستم خود:
```bash
cd /mnt/0D910DCF0D910DCF/work/research-few-shot
```
*(یا در ویندوز: `cd E:\work\research-few-shot`)*

### گام دوم: ایجاد و فعال‌سازی محیط مجازی پایتون (`.venv`)
**در لینوکس (Linux Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**در ویندوز (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### گام سوم: نصب وابستگی‌های نرم‌افزاری
پکیج‌های مورد نیاز در فایل [`requirements.txt`](../requirements.txt) مشخص شده‌اند:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## ۴. کتابخانه اسکریپت‌های اجرایی و دستورات CLI

پروژه به صورت کاملاً ماژولار و مبتنی بر آرگومان‌های خط فرمان (`argparse`) توسعه یافته است:

### ۱. اجرای پایپلاین اصلی ([`biodino_edge_pipeline.py`](../pipeline/biodino_edge_pipeline.py))

#### الف) اجرای بنچمارک استاندارد PlantVillage (۱۱ کلاس: سیب، انگور و سیب‌زمینی):
```bash
python pipeline/biodino_edge_pipeline.py --dataset plantvillage
```
* **خروجی:** ارزیابی دقت تک‌سید، آزمون ۳۰ اپیزودی مونت‌کارلو با فواصل اطمینان ۹۵٪، محاسبه نرخ منفی کاذب بالینی FNR و صدور گزارش طبقه‌بندی کامل.

#### ب) اجرای بنچمارک در شرایط واقعی مزرعه PlantDoc (۲۷ کلاس):
```bash
python pipeline/biodino_edge_pipeline.py --dataset plantdoc --k_shot 5
```
* **خروجی:** ارزیابی ۲۷ کلاسه تنش، ارزیابی ۵۰ اپیزود ۵ کلاسه ۵ شات، مقایسه با مدل VSF و ذخیره خروجی‌ها.

#### ج) اجرای همزمان هر دو بنچمارک و چاپ جدول مقایسه‌ای:
```bash
python pipeline/biodino_edge_pipeline.py --dataset both
```
* **خروجی:** اجرای خودکار روی هر دو دیتاست و چاپ جدول مقایسه‌ای درون‌ترمینالی و ذخیره در [`biodino_edge_results.json`](../biodino_edge_results.json).

#### د) سنجش تاخیر کوانتیزاسیون INT8 روی CPU:
```bash
python pipeline/biodino_edge_pipeline.py --dataset plantvillage --profile_latency
```

---

### ۲. اکسپورت مدل‌ها به فرمت ONNX و کوانتیزاسیون لبه ([`export_edge_models.py`](../pipeline/export_edge_models.py))

این اسکریپت مدل دو سطحی لبه را به فایل‌های کم‌حجم ONNX INT8 تبدیل و اعتبارسنجی می‌کند:
```bash
python pipeline/export_edge_models.py
```
* **خروجی‌ها:**
  - `biodino_mobile_int8.onnx` (مدل ۲۱ مگابایتی برای موبایل و رزبری‌پای)
  - `biodino_companion_int8.onnx` (مدل مکمل ۸۸ مگابایتی)
  - سینک خودکار فایل موبایل به پوشه `android_app/app/src/main/assets/` در صورت وجود پروژه اندروید.

---

### ۳. تست و استنتاج روی یک تک‌تصویر دلخواه ([`demo_predict.py`](../pipeline/demo_predict.py))
برای تست بلادرنگ و سریع یک تصویر بدون اجرای کل بنچمارک:
```bash
# تست روی یک فایل تصویر دلخواه
python pipeline/demo_predict.py --image "plantvillage/plantvillage dataset/color/Apple___Apple_scab/00075aa8-d81a-4184-8541-b692b78d398a___FREC_Scab 3335.JPG"

# تست خودکار روی یک نمونه تصادفی
python pipeline/demo_predict.py
```
* **خروجی:** نمایش زمان تاخیر استنتاج بر حسب میلی‌ثانیه، ضریب اطمینان گیت شانون، چارت میله‌ای اسکی درصدهای احتمال ۱۱ کلاس، و وضعیت نهایی سلامت یا بیماری برگ.

---

### ۴. تولید شکل‌های ۲ و ۳ مقاله: نقشه‌های سلف‌اتنشن و کالیبراسیون ECE ([`generate_saliency_and_calibration.py`](../pipeline/generate_saliency_and_calibration.py))
تولید شکل‌های آماده چاپ مقاله (PDF برداری و PNG با کیفیت ۳۰۰ DPI):
```bash
python pipeline/generate_saliency_and_calibration.py
```
* **خروجی‌ها:**
  - `paper/fig_saliency.pdf` و `paper/fig_saliency.png`
  - `paper/fig_calibration.pdf` و `paper/fig_calibration.png`

---

### ۵. تولید شکل ۱ مقاله: دیاگرام معماری ([`generate_architecture_diagram.py`](../pipeline/generate_architecture_diagram.py))
تولید دیاگرام برداری معماری BioDINO-Edge برای مقاله:
```bash
python pipeline/generate_architecture_diagram.py
```
* **خروجی‌ها:**
  - `paper/fig_architecture.pdf` و `paper/fig_architecture.png`

---

### ۶. اجرای نوت‌بوک تعاملی یکپارچه ژوپیتر (`pipeline/biodino_unified_pipeline.ipynb`)
برای مشاهده نمودارهای بصری، مراحل گام‌به‌گام و ماتریس درهم‌ریختگی رنگی ۱۱ کلاسه:
1. در محیط VS Code یا Jupyter Lab، فایل `pipeline/biodino_unified_pipeline.ipynb` را باز کنید.
2. سلول‌ها را به ترتیب اجرا نمایید تا هیت‌مپ‌های ماتریس سردرگمی رسم گردند.

---

## ۸. جدول هایپرپارامترهای کلیدی پایپلاین (`EdgeConfig`)

| پارامتر در `EdgeConfig` | مقدار پیش‌فرض | عملکرد فنی و منطق انتخاب |
| :--- | :---: | :--- |
| `seed` | `42` | بذر تصادفی ثابت جهت تضمین بازتولیدپذیری قطعی نتایج. |
| `k_shot` | `5` | تعداد نمونه‌های پشتیبان به ازای هر بیماری در سناریوی Few-Shot. |
| `tau_conf` | `0.35` | آستانه گیت مارجین-آنتروپی شانون؛ فریم‌های با اطمینان بالاتر مستقیماً از مسیر سریع DINOv2 در کمتر از ۲۲ میلی‌ثانیه رد می‌شوند. |
| `lda_ensemble_weight` | `0.75` | وزن فیشر در انسمبل خطی ($0.75 \times \text{LDA} + 0.25 \times \text{Text}$). |
| `text_fusion_weight` | `0.20` | وزن پروتوتایپ متنی در تلفیق BioCLIP ($0.80 \times \text{Vis} + 0.20 \times \text{Text}$). |
| `base_ensemble_weight` | `0.65` | ضریب تلفیق لاجیت پایه ($0.65 \times \text{BioCLIP} + 0.35 \times \text{DINOv2}$). |
| `logit_scale` | `20.0` | ضریب مقیاس‌گذاری شباهت کسینوسی قبل از اعمال تابع Softmax. |
| `support_views` | `8` | تعداد نماهای افزوده (۱ اصلی + ۷ آگمنتاسیون چنددیدگاهی). |
| `batch_size` | `128` | اندازه دسته‌های استنتاج کوئری برای بالاترین بهره‌وری پردازنده. |

---

## ۹. راهنمای عیب‌یابی و رفع خطاهای متداول (Troubleshooting)

| پیام خطا | دلیل فنی | راهکار حل قطعی |
| :--- | :--- | :--- |
| `CUDA out of memory` | حافظه کارت گرافیک برای لود همزمان مدل‌ها کافی نیست. | کد به صورت خودکار در صورت کمبود VRAM به CPU سوییچ می‌کند؛ یا می‌توانید بچ‌سایز را در `EdgeConfig` به ۳۲ یا ۶۴ کاهش دهید. |
| `FileNotFoundError: plantvillage` | مسیر تصاویر خام تغییر کرده است. | از وجود پوشه `plantvillage` در ریشه پروژه اطمینان حاصل کنید یا از فایل کَش ۲۷ مگابایتی `cache_artifacts/pv_novel_embeddings.pt` بهره ببرید. |
| `ModuleNotFoundError: open_clip` | کتابخانه BioCLIP نصب نیست. | دستور `pip install open_clip_torch timm` را در محیط فعال `.venv` اجرا کنید. |
| خطای فونت در پلات‌های مات‌پلات‌لیب | ناسازگاری فونت سیستم با کاراکترهای خاص. | اسکریپت‌ها از نام‌های استاندارد انگلیسی پاتوژن‌ها در نمودارها استفاده می‌کنند تا بدون نیاز به پکیج‌های فونت جانبی در لینوکس و ویندوز رندر شوند. |

---

> [!TIP]
> برای مطالعه عمیق تئوریک، به [راهنمای الگوریتمی (`algorithm_guide.md`)](algorithm_guide.md) و برای آموزش خط‌به‌خط ساختار کدها، به [آموزش جامع کدها (`code_walkthrough_guide.md`)](code_walkthrough_guide.md) مراجعه فرمایید.

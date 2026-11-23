# واژه‌نامه تخصصی و اصطلاح‌شناسی جامع BioDINO-Edge
### (Comprehensive Bilingual Terminology & Concise Glossary: English ➔ فارسی)

> **هدف این سند:**  
> این واژه‌نامه تخصصی شامل تمام اصطلاحات، مفاهیم ریاضی، تئوریک، الگوریتمی، مهندسی نرم‌افزار، لبه و پاتولوژی گیاهی استفاده‌شده در مقاله‌ها، کدهای منبع و اسناد پروژه **BioDINO-Edge** است. برای هر اصطلاح، نام انگلیسی، معادل استاندارد فارسی، و تعریفی کوتاه و چندکلمه‌ای (Concise Definition) به دو زبان فارسی و انگلیسی درج شده است.

---

## فهرست دسته‌بندی موضوعی اصطلاحات

1. [معماری مدل‌ها و یادگیری عمیق (Model Architecture & Deep Learning)](#۱-معماری-مدل‌ها-و-یادگیری-عمیق-model-architecture--deep-learning)
2. [یادگیری اندک‌نمونه و هندسه منیفولد (Few-Shot Learning & Manifold Geometry)](#۲-یادگیری-اندک‌نمونه-و-هندسه-منیفولد-few-shot-learning--manifold-geometry)
3. [آمار تحلیلی و تفکیک خطی فیشر (Statistics & Fisher-LDA Alignment)](#۳-آمار-تحلیلی-و-تفکیک-خطی-فیشر-statistics--fisher-lda-alignment)
4. [تئوری اطلاعات، عدم قطعیت و گیتینگ (Information Theory & Uncertainty Gating)](#۴-تئوری-اطلاعات-عدم-قطعیت-و-گیتینگ-information-theory--uncertainty-gating)
5. [پردازش زبان، پرامپتینگ و تبارشناسی (NLP, Taxonomic Prompts & Biometrics)](#۵-پردازش-زبان-پرامپتینگ-و-تبارشناسی-nlp-taxonomic-prompts--biometrics)
6. [مهندسی لبه، موبایل و کوانتیزاسیون (Edge Deployment, Mobile & Quantization)](#۶-مهندسی-لبه-موبایل-و-کوانتیزاسیون-edge-deployment-mobile--quantization)
7. [پاتولوژی گیاهی و واژگان بیماری‌شناسی (Plant Pathology & Botanical Terms)](#۷-پاتولوژی-گیاهی-و-واژگان-بیماری‌شناسی-plant-pathology--botanical-terms)
8. [معیارهای ارزیابی و پروتکل‌های آماری (Benchmarking & Evaluation Metrics)](#۸-معیارهای-ارزیابی-و-پروتکل‌های-آماری-benchmarking--evaluation-metrics)

---

### ۱. معماری مدل‌ها و یادگیری عمیق (Model Architecture & Deep Learning)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۱ | **Vision Transformer (ViT)** | ترنسفورمر بینایی | معماری پردازش تصویر بر پایه پچ‌ها و مکانیزم خودتوجهی | Image architecture based on patches and self-attention |
| ۲ | **DINOv2** | دینو نگارش ۲ | مدل خودنظارتی متمرکز بر ویژگی‌های محلی و بافت میکروسکوپی | Self-supervised model capturing dense local patch features |
| ۳ | **BioCLIP** | بایوکلیپ | مدل چندوجهی بینایی-زبانی آموزش‌دیده روی داده‌های زیستی | Vision-language foundation model trained on biology datasets |
| ۴ | **Self-Attention** | توجه خودکار | مکانیزم وزن‌دهی ارتباط بین تمام پچ‌های تصویر با یکدیگر | Weighting mechanism capturing pairwise token interactions |
| ۵ | **Multi-Head Self-Attention (MHSA)** | خودتوجهی چندسر | موازی‌سازی توجه خودکار در چندین زیرفضای بازنمایی مجزا | Parallel attention heads projecting into distinct subspaces |
| ۶ | **Backbone / Feature Extractor** | ستون‌فقرات / استخراج‌کننده ویژگی | شبکه عصبی پایه استخراج‌کننده بردار ویژگی غنی از ورودی خام | Base neural network extracting rich feature embeddings |
| ۷ | **Frozen Weights** | وزن‌های منجمد / فریز شده | وزن‌های شبکه‌ای که آموزش ندیده و گرادیان آن‌ها صفر است | Fixed model parameters receiving zero gradient updates |
| ۸ | **Patch Embedding** | جاسازی وصله‌ای | تقسیم تصویر به قطعات مربعی و تصویر خطی آن‌ها به بردار | Dividing image into patches and linearly projecting to vectors |
| ۹ | **Class Token (`[CLS]`)** | توکن کلاس | بردار یادگیرنده‌ای که نمای کلی تمام تصویر را تجمیع می‌کند | Learnable token aggregating global image-level semantics |
| ۱۰ | **Positional Embedding** | جاسازی موقعیت | بردار نشان‌دهنده مختصات مکانی هر پچ در صفحه تصویر | Spatial vectors encoding 2D patch coordinates in an image |
| ۱۱ | **Dual-Backbone Architecture** | معماری دو ستون‌فقراتی | ترکیب دو استخراج‌کننده ویژگی ناهمگن برای پوشش چندمقیاسی | Pairing two heterogeneous encoders for multi-scale cues |
| ۱۲ | **Dynamic Image Size** | اندازه تصویر پویا | پذیرش ابعاد دلخواه ورودی با درون‌یابی موقعیت مکانی پچ‌ها | Interpolating positional embeddings for arbitrary resolutions |
| ۱۳ | **Patch Size vs. Grid ($P$ vs $G$)** | ابعاد پچ در برابر گرید | اندازه پیکسلی پچ ($14\times14$ در DINOv2 و $16\times16$ در BioCLIP) در برابر گرید تصویر | Patch pixel size ($14\times14$ in DINOv2, $16\times16$ in BioCLIP) vs 2D token grid |

---

### ۲. یادگیری اندک‌نمونه و هندسه منیفولد (Few-Shot Learning & Manifold Geometry)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۱۳ | **Few-Shot Learning (FSL)** | یادگیری اندک‌نمونه | دسته‌بندی کلاس‌های نادیده تنها با تعداد انگشت‌شماری نمونه | Classifying novel categories from only a few labeled examples |
| ۱۴ | **Support Set ($\mathcal{S}$)** | مجموعه پشتیبان | نمونه‌های برچسب‌دار اندک استفاده‌شده برای شناسایی کلاس | Few labeled examples guiding novel class characterization |
| ۱۵ | **Query Set ($\mathcal{Q}$)** | مجموعه کوئری / پرسمان | نمونه‌های بدون‌برچسب که باید در زمان آزمون رده‌بندی شوند | Unlabeled test samples evaluated during episodic inference |
| ۱۶ | **$N$-Way $K$-Shot** | رژیم $N$ رده با $K$ نمونه | دسته‌بندی بین $N$ کلاس با $K$ تصویر آموزشی به ازای هر کلاس | Classification task across $N$ classes using $K$ samples each |
| ۱۷ | **Class Prototype ($\boldsymbol{\mu}_c$)** | پروتوتایپ / مرکزوار کلاس | بردار میانگین نمونه‌های یک کلاس در فضای ویژگی | Mean embedding vector representing a specific class cluster |
| ۱۸ | **Fréchet Mean** | مرکز ثقل فرشه | تعمیم مرکز جرم هندسی به فضاهای خمیده ریمانی | Generalization of centroids to Riemannian curved manifolds |
| ۱۹ | **Hilbert Space ($\mathcal{H}$)** | فضای هیلبرت | فضای برداری کامل مجهز به ضرب داخلی و مفهوم زاویه | Complete vector space endowed with an inner product |
| ۲۰ | **Direct Sum ($\oplus$)** | جمع مستقیم زیرفضاها | الحاق دو فضای برداری بدون تداخل و همپوشانی ابعاد | Concatenating vector spaces preserving orthogonal dimensions |
| ۲۱ | **Unit Hypersphere ($\mathbb{S}^{d-1}$)** | ابرکره واحد | پوسته کروی با شعاع ۱ که در آن اندازه طول بردارها برابر است | Manifold of vectors constrained to unit Euclidean norm |
| ۲۲ | **Cosine Similarity** | شباهت کسینوسی | کسینوس زاویه بین دو بردار که مستقل از طول آن‌هاست | Cosine of angle between vectors, invariant to scale |
| ۲۳ | **$\ell_2$ Normalization** | نرمال‌سازی اقلیدسی | تقسیم بردار بر اندازه طولش برای رسیدن به طول واحد ۱ | Dividing vector by its magnitude to ensure unit length |
| ۲۴ | **Inductive Learning** | یادگیری استقرایی | استنتاج بر روی هر نمونه آزمون کاملاً مستقل از سایر نمونه‌ها | Classifying test queries purely independently one by one |
| ۲۵ | **Transductive Inference** | استنتاج فراالقایی | استفاده مشترک از هندسه کل داده‌های تست برای بهبود تصمیم | Leveraging unlabeled query batch geometry for better decision |
| ۲۶ | **Pseudo-Labeling** | شبه‌برچسب‌گذاری | تخصیص برچسب موقت به داده‌های تست بااطمینان بالا | Assigning high-confidence predicted labels to test samples |
| ۲۷ | **High-Confidence Anchors** | انکورها / نمونه‌های لنگرگاهی | داده‌های تستی که اطمینان بسیار بالا دارند و مبنای اصلاح منیفولدند | High-certainty query samples used to anchor manifold shifts |

---

### ۳. آمار تحلیلی و تفکیک خطی فیشر (Statistics & Fisher-LDA Alignment)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۲۸ | **Fisher-LDA** | تحلیل تفکیک خطی فیشر | تصویر خطی برای حداکثر کردن فاصله کلاس‌ها و حداقل کردن پراکندگی درون‌کلاسی | Linear projection maximizing between-class scatter over within-class |
| ۲۹ | **Within-Class Scatter ($\mathbf{S}_W$)** | پراکندگی درون‌کلاسی | ماتریس واریانس نمونه‌های هر کلاس حول مرکزوار خود | Covariance matrix measuring sample spread around class means |
| ۳۰ | **Between-Class Scatter ($\mathbf{S}_B$)** | پراکندگی بین‌کلاسی | ماتریس بیانگر فاصله مراکز کلاس‌ها از میانگین کل | Covariance matrix measuring separation between class centroids |
| ۳۱ | **HDLSS Problem** | چالش ابعاد بالا و نمونه کم | شرایطی که تعداد ویژگی‌ها بسیار بیشتر از تعداد نمونه‌هاست ($d \gg n$) | High-dimension low-sample-size setting causing singular matrices |
| ۳۲ | **Singular Matrix** | ماتریس تکین | ماتریسی با دترمینان صفر که معکوس‌پذیر نیست | Matrix with zero determinant lacking an algebraic inverse |
| ۳۳ | **Ledoit-Wolf Shrinkage** | انقباض لدویت-ولف | ترکیب تحلیلی کوواریانس تجربی با ماتریس همانی برای معکوس‌پذیری | Regularizing empirical covariance with identity analytically |
| ۳۴ | **Shrinkage Intensity ($\rho^*$)** | ضریب بهینه انقباض | عددی در بازه $[0, 1]$ که تعادل بین واریانس و بایاس را کنترل می‌کند | Optimal scalar minimizing Frobenius mean squared error |
| ۳۵ | **Frobenius Norm ($\|\cdot\|_F$)** | نُرم فروبنیوس | ریشه دوم مجموع مربعات تمام عناصر یک ماتریس | Square root of the sum of squared elements in a matrix |
| ۳۶ | **LSQR Solver** | سالور کمترین مربعات کیو‌آر | الگوریتم تکراری حل دستگاه‌های معادلات خطی بزرگ و ماتریس‌های خلوت | Iterative method solving large-scale regularized linear systems |
| ۳۷ | **Temperature Scaling ($T, \beta$)** | مقیاس‌بندی دمایی لوجیت | تقسیم یا ضرب لوجیت‌ها برای تنظیم تیزی توزیع سافت‌مکس | Calibrating logit dispersion to tune softmax confidence sharpness |
| ۳۸ | **Closed-Form Solution** | پاسخ فرم‌بسته / تحلیلی | راه‌حل ریاضی دقیق بدون نیاز به بهینه‌سازی گرادیانی تکرارشونده | Exact mathematical solution derived without iterative loops |

---

### ۴. تئوری اطلاعات، عدم قطعیت و گیتینگ (Information Theory & Uncertainty Gating)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۳۹ | **Shannon Entropy ($H(p)$)** | آنتروپی شانون | معیار اندازه‌گیری میزان عدم‌قطعیت و تصادفی بودن توزیع احتمال | Measure of uncertainty and chaos in a probability distribution |
| ۴۰ | **Normalized Entropy ($\tilde{H}$)** | آنتروپی نرمال‌شده | تقسیم آنتروپی بر $\ln(K)$ تا مقادیر همیشه در بازه $[0, 1]$ باشند | Entropy scaled by $\ln(K)$ to strictly bound values in $[0, 1]$ |
| ۴۱ | **Prediction Margin ($\Delta p$)** | مارجین تفاضلی پیش‌بینی | اختلاف احتمال بین کلاس رتبه اول و کلاس رتبه دوم | Probability gap between Top-1 and Top-2 ranked classes |
| ۴۲ | **Shannon Composite Confidence** | شاخص اطمینان ترکیبی شانون | حاصل‌ضرب تفاضل مارجین در متمم آنتروپی نرمال‌شده ($\Delta p \cdot (1 - \tilde{H})$) | Margin multiplied by entropy complement filtering uncertain queries |
| ۴۳ | **Uncertainty Gate** | گیت عدم‌قطعیت | فیلتر تصمیم‌گیری برای پذیرش یا بازبینی پاسخ مدل | Decision boundary gating predictions based on confidence |
| ۴۴ | **Out-of-Distribution (OOD)** | داده خارج از توزیع | ورودی‌هایی که با توزیع داده‌های آموزش مطابقت ندارند (برگ نامربوط، تاریکی) | Inputs outside the training distribution domain |
| ۴۵ | **Foliage Gate** | گیت منیفولد شاخ‌وبرگ | فیلتر شباهت کسینوسی برای حذف اشیای غیربرگ و صفحات سیاه | Cosine manifold filter rejecting non-leaf objects or dark frames |
| ۴۶ | **Cascaded Routing** | مسیریابی آبشاری | هدایت نمونه‌های واضح به مدل سریع و نمونه‌های مبهم به مدل سنگین | Routing confident queries to fast tier and hard cases to deep tier |
| ۴۷ | **Early Exit / Fast Path** | خروج زودهنگام / مسیر سریع | صدور پیش‌بینی با مدل سبک در زمان اندک بدون درگیرسازی مدل‌های دوم | Rapid prediction emission skipping companion model computation |
| ۴۸ | **Expected Calibration Error (ECE)** | خطای کالیبراسیون مورد انتظار | اختلاف میانگین بین اطمینان اعلام‌شده مدل و دقت تجربی واقعی | Difference between model confidence and empirical accuracy |
| ۴۹ | **Reliability Diagram** | نمودار پایایی | پلات ارزیابی تطابق بین درصدهای اطمینان و نرخ صحت پیش‌بینی | Diagram mapping confidence buckets against true empirical accuracy |

---

### ۵. پردازش زبان، پرامپتینگ و تبارشناسی (NLP, Taxonomic Prompts & Biometrics)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۵۰ | **Taxonomic Prompt** | پرامپت تکسونومیک / تبارشناسی | توصیف متنی دقیق شامل نام علمی گیاه، پاتوژن و مشخصات بالینی | Expert text prompt coupling plant genus, species, and pathogen |
| ۵۱ | **Syntactic Negation Poisoning** | مسمومیت نحوی نفی | انحراف اتنشن ترنسفورمر به دلیل وجود کلمات منفی نظیر "بدون لکه" | Attention distortion caused by negative tokens like "no spots" |
| ۵۲ | **Positive Taxonomic Priors** | پیش‌دانش‌های ایجابی تکسونومیک | استفاده از توصیفات مثبت بالینی به‌جای کلمات نفی | Affirmative morphological descriptors avoiding negation terms |
| ۵۳ | **Zero-Shot Synthesis** | سنتز بدون‌نمونه | ساخت پروتوتایپ کلاسی فقط با متن و بدون حتی یک عکس آموزشی | Creating class prototypes from text alone with zero images |
| ۵۴ | **Multi-Prompt Ensembling** | هم‌افزایی چندپرامپتی | میانگین‌گیری از بردارهای چند جمله توصیفی برای خنثی‌سازی نویز نحوی | Averaging embeddings from multiple prompts to cancel linguistic noise |
| ۵۵ | **Tokenizer** | توکن‌ساز | ابزار تجزیه رشته متنی به شناسه‌های عددی قابل پردازش در ترنسفورمر | Parser splitting text strings into numerical input tokens |
| ۵۶ | **Cross-Modal Alignment** | انطباق متقابل چندوجهی | نگاشت بردارهای تصویر و متن به یک فضای برداری مشترک هم‌بُعد | Projecting visual and textual features into a shared manifold |

---

### ۶. مهندسی لبه، موبایل و کوانتیزاسیون (Edge Deployment, Mobile & Quantization)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۵۷ | **Edge Computing** | محاسبات لبه | پردازش داده‌ها مستقیماً روی دستگاه هوشمند محلی بدون نیاز به اینترنت | On-device data processing near the source without cloud reliance |
| ۵۸ | **ONNX Runtime (ORT)** | موتور اجرای اونکس | فریم‌ورک متن‌باز با کارایی بالا برای اجرای بهینه مدل‌های عصبی | Cross-platform high-performance engine running neural graphs |
| ۵۹ | **Dynamic Quantization** | کوانتیزاسیون پویا | تبدیل وزن‌های ممیز شناور (FP32) به اعداد صحیح (INT8) | Compressing FP32 matrix weights into 8-bit integers dynamically |
| ۶۰ | **Symmetric Quantization** | کوانتیزاسیون متقارن | نگاشت دامنه‌های مثبت و منفی با نقطه صفر دقیقاً در مرکز ($Z=0$) | Quantization mapping around zero origin without zero-point shift |
| ۶۱ | **Quantization Scale ($S$)** | ضریب مقیاس کوانتیزاسیون | فاکتور تبدیل بازه پیوسته اعشاری به بازه گسسته عددی صحیح | Multiplier mapping continuous float range into discrete integer steps |
| ۶۲ | **NNAPI** | ای‌پی‌آی شبکه عصبی اندروید | واسط نرم‌افزاری شتاب‌دهنده سخت‌افزاری NPU/GPU در اندروید | Android API accessing hardware accelerators like NPU and GPU |
| ۶۳ | **Preallocated Tensor** | تانسور از پیش‌تخصیص‌یافته | رزرو فضای حافظه بافر قبل از استنتاج برای حذف تاخیر زباله‌روب | Reused memory buffer eliminating runtime garbage collection pauses |
| ۶۴ | **CameraX Pipeline** | پایپلاین کمراایکس | کتابخانه اندروید جت‌پک برای پردازش بلادرنگ فریم‌های دوربین | Jetpack library managing live camera feeds without frame drops |
| ۶۵ | **Latency ($ms$)** | تاخیر زمانی استنتاج | مدت‌زمان پردازش یک تصویر از ورودی تا صدور پیش‌بینی به میلی‌ثانیه | Milliseconds elapsed during a complete model inference pass |
| ۶۶ | **Throughput (FPS)** | توان عملیاتی فریم بر ثانیه | تعداد فریم‌هایی که سیستم می‌تواند در هر ثانیه پردازش کند | Number of processed video frames completed per second |
| ۶۷ | **ARM NEON / SDOT** | ثبات‌های برداری آرم | دستورالعمل‌های سخت‌افزاری پردازنده موبایل برای ضرب ۴ بردار INT8 همزمان | Hardware vector instructions accelerating 4-way INT8 dot products |
| ۶۸ | **Tier 1 Fast Model** | مدل سبک سطح اول | مدل مستقل DINOv2 با حجم ۲۱ مگابایت مقیم در اپلیکیشن موبایل | Standalone 21 MB quantized on-device APK model |
| ۶۹ | **Tier 2 Companion Model** | مدل مکمل سطح دوم | مدل سنگین‌تر BioCLIP با حجم ۸۸ مگابایت برای رفع موارد مبهم | Heavier 88 MB on-demand companion resolving difficult ambiguities |

---

### ۷. پاتولوژی گیاهی و واژگان بیماری‌شناسی (Plant Pathology & Botanical Terms)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۷۰ | **Phytopathology** | گیاه‌پزشکی / پاتولوژی گیاهی | دانش مطالعه بیماری‌های گیاهی و عوامل بیماری‌زای آن‌ها | Scientific study of plant diseases caused by pathogens |
| ۷۱ | **Etiology** | سبب‌شناسی بیماری | بررسی منشأ زیستی، چرخه حیات و علل بروز بیماری گیاهی | Cause and biological origin of plant diseases |
| ۷۲ | **Pathogen** | پاتوژن / عامل بیماری‌زا | ارگانیسم زنده (قارچ، باکتری، ویروس) که موجب بیماری گیاه می‌شود | Biological organism causing infectious disease in hosts |
| ۷۳ | **Chlorosis** | کلروز / زردی بافت | زرد شدن برگ بر اثر فقدان، تخریب یا مهار تولید سبزینه (کلروفیل) | Yellowing of plant foliage due to chlorophyll insufficiency |
| ۷۴ | **Necrosis** | نکروز / بافت‌مردگی | مرگ موضعی و قهوه‌ای/سیاه شدن سلول‌ها و بافت‌های گیاهی | Premature cell death causing browned and dried plant tissue |
| ۷۵ | **Lesion** | ضایعه / لکه پاتولوژیک | ناحیه غیرعادی آسیب‌دیده بافت برگ ناشی از حمله بیماری | Localized structural foliage injury caused by disease attack |
| ۷۶ | **Apple Scab** | لکه سیاه سیب | بیماری قارچی با لکه‌های مخملی زیتونی (*Venturia inaequalis*) | Fungal foliage lesions caused by *Venturia inaequalis* |
| ۷۷ | **Frogeye Leaf Spot / Black Rot** | پوسیدگی سیاه / لکه چشم قورباغه‌ای | لکه‌های دایره‌ای با مرکز خرمایی و حاشیه تیره (*Botryosphaeria obtusa*) | Concentric leaf spots caused by *Botryosphaeria obtusa* |
| ۷۸ | **Cedar Apple Rust** | زنگ ارس سیب | لکه‌های نارنجی-زرد براق با جوش‌های زرد قارچی (*Gymnosporangium*) | Bright yellow-orange leaf aecia caused by *Gymnosporangium* |
| ۷۹ | **Grape Esca (Black Measles)** | عارضه اسکای انگور | کلروز و نکروز بین رگبرگی که خطوط ببری ایجاد می‌کند (*Phaeomoniella*) | Tiger-stripe foliar necrosis caused by vascular Esca fungi |
| ۸۰ | **Isariopsis Leaf Blight** | سوختگی برگی ایزاریوپسیس | لکه‌های قهوه‌ای زاویه‌دار محدود به رگبرگ‌ها (*Pseudocercospora vitis*) | Angular foliar lesions bounded by veinlets (*Pseudocercospora*) |
| ۸۱ | **Early Blight** | سوختگی زودرس | لکه‌های تیره با دوایر متحدالمرکز صفحه‌هدف (*Alternaria solani*) | Concentric target-board ring lesions caused by *Alternaria solani* |
| ۸۲ | **Late Blight** | سوختگی دیررس | ضایعات آب‌سوخته و گسترش‌یابنده با لبه سفیدک قارچی (*Phytophthora*) | Water-soaked greasy foliar decay caused by *Phytophthora infestans* |
| ۸۳ | **Tiger-Stripe Foliar Pattern** | الگوی راه‌راه ببری برگ | نوارهای زرد و قهوه‌ای نکروز بین رگبرگ‌های سبز سالم در اسکا | Yellow-necrotic bands flanking green veins characteristic of Esca |
| ۸۴ | **Target-Board Concentric Rings** | دوایر متحدالمرکز آماجگاهی | حلقه‌های تودرتو مانند سیبل تیراندازی در لکه‌های آلترناریا | Bullseye ring morphology typical of Alternaria fungal blights |

---

### ۸. معیارهای ارزیابی و پروتکل‌های آماری (Benchmarking & Evaluation Metrics)

| # | اصطلاح انگلیسی (English Term) | معادل فارسی | تعریف کوتاه فارسی (Concise Definition) | English Short Definition |
| :-: | :--- | :--- | :--- | :--- |
| ۸۵ | **False Negative (FN)** | منفی کاذب | طبقه‌بندی اشتباه یک برگ آلوده به عنوان برگ سالم | Misclassifying a diseased leaf as healthy (dangerous error) |
| ۸۶ | **False Negative Rate (FNR)** | نرخ منفی کاذب | درصد نمونه‌های آلوده‌ای که توسط مدل سالم گزارش شده‌اند | Percentage of infected leaves wrongly predicted as healthy |
| ۸۷ | **Macro-Averaged F1-Score** | شاخص اف۱ میانگین کلان | میانگین هارمونیک دقت و بازخوانی با وزن برابر برای تمام کلاس‌ها | Unweighted mean of harmonic precision-recall across all classes |
| ۸۸ | **Monte Carlo Benchmark** | بنچمارک مونت‌کارلو | ارزیابی دقت در چندین اپیزود مستقل با سیدهای تصادفی مختلف | Repeating evaluation over dozens of randomized data splits |
| ۸۹ | **Confidence Interval (95% CI)** | فاصله اطمینان ۹۵٪ | بازه آماری که با اطمینان ۹۵ درصد میانگین واقعی در آن قرار دارد | Margin of error bounding the true population mean ($1.96 \cdot \frac{\sigma}{\sqrt{N}}$) |
| ۹۰ | **Confusion Matrix** | ماتریس درهم‌ریختگی | جدول مقایسه فراوانی برچسب‌های واقعی در برابر پیش‌بینی‌ها | Contingency table contrasting actual classes against predictions |
| ۹۱ | **Zero Data Leakage** | عدم نشت داده | ایزولاسیون مطلق داده‌های آموزش و پشتیبان از نمونه‌های آزمون | Absolute separation of support splits from test query pools |
| ۹۲ | **Cross-Dataset Generalization** | تعمیم‌پذیری بین‌دیتاستی | ارزیابی مدل آموزش‌دیده روی یک دیتاست بر روی دیتاست کاملاً مجزا | Evaluating performance across completely distinct dataset distributions |
| ۹۳ | **Saliency Map** | نقشه برجستگی بصری | تصویر میزان توجه و تمرکز مکانیزم اتنشن بر روی لکه‌های بیماری | Heatmap visualizing model attention intensity across image pixels |
| ۹۴ | **Peak Accuracy** | دقت قله / بیشینه | بالاترین دقت ثبت‌شده در بهترین اپیزود شبیه‌سازی | Highest empirical accuracy recorded among all evaluated splits |
| ۹۵ | **PlantVillage** | دیتاست پلنت‌ویلیج | دیتاست استاندارد آزمایشگاهی با پس‌زمینه تک‌رنگ یکدست | Standard laboratory plant leaf benchmark with plain backgrounds |
| ۹۶ | **PlantDoc** | دیتاست پلنت‌داک | دیتاست چالشی تصاویر طبیعی مزرعه با پس‌زمینه‌های شلوغ و نویزی | Challenging real-world field plant pathology benchmark |
| ۹۷ | **Maximum Calibration Error (MCE)** | بیشترین خطای کالیبراسیون | حداکثر اختلاف بین دقت و اطمینان در بحرانی‌ترین سطل آماری | Worst-case calibration gap observed across all evaluated bins |
| ۹۸ | **Calibration Gap** | شکاف کالیبراسیون | قدرمطلق تفاضل بین دقت تجربی واقعی و میانگین اطمینان پیش‌بینی‌شده | Absolute difference between empirical accuracy and mean confidence |
| ۹۹ | **Underconfidence** | کم‌اطمینانی مفرط | شرایطی که مدل دقت بالایی دارد اما احتمالات پایینی صادر می‌کند | Regime where model accuracy strictly exceeds predicted probability |
| ۱۰۰ | **Overconfidence** | بیش‌اطمینانی کاذب | شرایطی که مدل با اطمینان ۹۹٪ یک پاسخ کاملاً اشتباه صادر می‌کند | Regime where predicted probability dangerously exceeds accuracy |
| ۱۰۱ | **Peak Logit Dilution** | رقیق‌شدگی لوجیت بیشینه | افت احتمال کلاس برنده در اثر ادغام خطی با پرامپت‌های با آنتروپی بالا | Softening of max logit caused by convex mixing with diffuse posteriors |

---

## ماتریس اتصال سریع مفاهیم به فایل‌های پروژه (File Cross-Reference)

| رده مفهومی | اصطلاحات شاخص | فایل پیاده‌سازی یا سند تحلیلی |
| :--- | :--- | :--- |
| **معماری و پایپلاین** | DINOv2, BioCLIP, ViT, Dual-Backbone | [`biodino_edge_pipeline.py`](../biodino_edge_pipeline.py) \| [`code_math_algorithm_deep_dive.md`](code_math_algorithm_deep_dive.md) |
| **ریاضیات و فیشر** | Hilbert Space, Ledoit-Wolf, $\mathbf{S}_W, \mathbf{S}_B$ | [`algorithm_guide.md`](algorithm_guide.md) \| [`mathematics_foundation_guide.md`](mathematics_foundation_guide.md) |
| **گیتینگ و عدم‌قطعیت** | Shannon Entropy, Margin, Confidence, OOD | [`biodino_edge_pipeline.py`](../biodino_edge_pipeline.py) \| [`failure_modes_and_robustness.md`](failure_modes_and_robustness.md) |
| **پرامپتینگ زیستی** | Taxonomic Prompts, Negation Poisoning | [`taxonomic_prompts.py`](../taxonomic_prompts.py) \| [`plant_pathology_domain_guide.md`](plant_pathology_domain_guide.md) |
| **لبه و موبایل** | ONNX Runtime, INT8 Quantization, NNAPI, CameraX | [`export_edge_models.py`](../export_edge_models.py) \| [`android_mobile_app_guide.md`](android_mobile_app_guide.md) |
| **پاتولوژی گیاهی** | Venturia, Esca, Alternaria, Phytophthora | [`taxonomic_prompts.py`](../taxonomic_prompts.py) \| [`plant_pathology_domain_guide.md`](plant_pathology_domain_guide.md) |
| **ارزیابی آماری** | Monte Carlo, 95% CI, Macro F1, FN Rate | [`biodino_edge_pipeline.py`](../biodino_edge_pipeline.py) \| [`reviewer_rebuttal_playbook.md`](reviewer_rebuttal_playbook.md) |


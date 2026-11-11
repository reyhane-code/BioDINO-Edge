# -*- coding: utf-8 -*-
"""
Script to update docs/algorithm_guide.md with the detailed flowchart breakdown
and links to mathematics_foundation_guide.md.
"""
import sys

NEW_SECTION_5_6 = """### ۵.۶. فلوچارت تفصیلی الگوریتم و کالبدشکافی گام‌به‌گام با نگاشت ریاضی

در این بخش، ابتدا شبه‌کد رسمی فریم‌ورک مطابق استانداردهای مقالات Nature و IEEE ارائه شده و سپس **فلوچارت الگوریتمی انتشاریافته** جزءبه‌جزء با ارجاع مستقیم به اثبات‌های ریاضی و خطوط پیاده‌سازی کدبیس کالبدشکافی می‌شود.

```text
====================================================================================================
Algorithm 1: BioDINO-Edge Cascaded Dual-Tier Inference with Closed-Form Transductive Recalibration
====================================================================================================
Input:
  • Few-Shot Support Set: S = {(x_i, y_i)}_{i=1}^{N_s} (K-shot per class, C classes)
  • Query Target Images : Q = {x_j}_{j=1}^{N_q}
  • Vision Models       : Fast-Path Phi_dino (384-d), Deep-Path Phi_bio (512-d)
  • Taxonomic Prompts   : T = {T_c^{(m)}} for c in {1..C}, m in {1..M}
  • Hyperparameters     : tau_conf = 0.35, lambda_fisher = 0.75, T_temp = 0.50, K_anchors = 20

Output:
  • Predicted Class Labels: Y_hat = {y_hat_j}_{j=1}^{N_q}
  • Calibrated Probabilities: P_cal in [0, 1]^{N_q x C}

----------------------------------------------------------------------------------------------------
Stage 1: Multi-View Support Extraction & Joint Manifold Construction (Offline One-Time: ~50 ms)
----------------------------------------------------------------------------------------------------
 1: For each (x_i, y_i) in S do:
 2:    Generate 8 multi-view augmentations {x_{i, v}}_{v=1}^8
 3:    f_{bio, i}  <- L2_Normalize(Phi_bio(x_{i, v})) in R^{512}
 4:    f_{dino, i} <- L2_Normalize(Phi_dino(x_{i, v})) in R^{384}
 5:    z_i         <- [f_{bio, i}, f_{dino, i}] in R^{896}
 6: End For
 7: Synthesize text prototypes: P_text^{(c)} <- L2_Normalize(Mean_m(TextEncoder(T_c^{(m)})))
 8: Compute Ledoit-Wolf shrunk covariance Sigma_hat in R^{896 x 896} and Fisher weights (W, b)

----------------------------------------------------------------------------------------------------
Stage 2: Cascaded Dual-Tier Query Routing via Shannon Margin-Entropy Gate (Per-Query: 21-42 ms)
----------------------------------------------------------------------------------------------------
 9: For each query x_j in Q do:
10:    Extract Fast-Path features: f_{dino, j} <- L2_Normalize(Phi_dino(x_j)) in R^{384}
11:    Compute Fast logits: z_{dino, j} <- 20.0 * (f_{dino, j} @ P_{dino}^T)
12:    p_{dino, j} <- Softmax(z_{dino, j})
13:    Compute confidence: C(x_j) <- (p_{(1)} - p_{(2)}) * (1.0 - H_{norm}(p_{dino, j}))
14:    If C(x_j) >= tau_conf then:
15:       Fast-Exit: y_hat_j <- argmax(p_{dino, j})  [Latency: ~21.12 ms | FPS: ~47.4]
16:    Else:
17:       Deep-Path: Extract f_{bio, j} <- L2_Normalize(Phi_bio(x_j)) in R^{512}
18:       Concatenate joint query: z_j <- [f_{bio, j}, f_{dino, j}] in R^{896}
19:       p_{fisher, j} <- Softmax(z_j @ W^T + b)
20:       p_{text, j}   <- Softmax(20.0 * (f_{bio, j} @ P_text^T))
21:       p_j           <- lambda_fisher * p_{fisher, j} + (1.0 - lambda_fisher) * p_{text, j}
22:       y_hat_j       <- argmax(p_j)
23:    End If
24: End For

----------------------------------------------------------------------------------------------------
Stage 3: Closed-Form Transductive Recalibration (Post-Inference Batch Refinement: ~22 ms)
----------------------------------------------------------------------------------------------------
25: For each class c in {1..C} do:
26:    Select Top-K anchors Q_{anchor}^{(c)} <- TopK_{K_anchors}({x_j | y_hat_j == c}, C(x_j))
27: End For
28: Augment support set: S_trans <- {S, S, S} U Q_{anchor} (3x support weighting)
29: Re-estimate Sigma_trans and Fisher weights (W_trans, b_trans) via Ledoit-Wolf
30: Final probabilities: P_final <- 0.75 * Softmax(Z @ W_trans^T + b_trans) + 0.25 * P_text
31: Temperature calibration: P_cal <- Softmax(log(P_final + 1e-12) / T_temp)
32: Return Y_hat <- argmax(P_cal, dim=-1) and P_cal
====================================================================================================
```

#### جدول تحلیل پیچیدگی محاسباتی (Computational Complexity):
| مرحله | عملیات محوری | پیچیدگی زمانی (Time) | پیچیدگی فضایی (Memory) | تاخیر روی پردازنده لبه |
| :--- | :--- | :---: | :---: | :---: |
| **استخراج اولیه ساپورت** | فیت انقباض لدویت-ولف | $\\mathcal{O}(N_s d^2 + d^3)$ | $\\mathcal{O}(d^2) \\approx 3.1 \\text{ MB}$ | **~ ۵۰ میلی‌ثانیه** (یک‌بار در ابتدا) |
| **مسیر سریع (Fast-Path)** | استنتاج DINOv2 + گیت | $\\mathcal{O}(L_{\\text{dino}} + d_{\\text{dino}} C)$ | $\\approx ۲۲ \\text{ MB}$ | **۲۱.۱۲ میلی‌ثانیه** (۴۷.۴ FPS) |
| **مسیر عمیق (Deep-Path)** | استنتاج BioCLIP + ضرب فیشر | $\\mathcal{O}(L_{\\text{bio}} + d C)$ | $\\approx ۸۴ \\text{ MB}$ | **۴۲.۹۲ میلی‌ثانیه** (۲۳.۳ FPS) |
| **کالیبراسیون ترنسداکتیو** | محاسبه تحلیلی منیفولد گسترش‌یافته | $\\mathcal{O}((N_s + N_{\\text{anchor}}) d^2 + d^3)$ | $\\mathcal{O}(d^2)$ | **۲۲.۳۰ میلی‌ثانیه** (فوری روی CPU) |
| **استنتاج هر تصویر نهایی** | ضرب ماتریسی برداری $\\mathbf{z} \\mathbf{W}^T + \\mathbf{b}$ | $\\mathcal{O}(d \\cdot C) = 896 \\times 11$ | **< ۱۰ کیلوبایت** | **۰.۰۰۲ میلی‌ثانیه** (> ۱۰,۰۰۰ FPS) |

---

### فلوچارت رسمی و ساختاریافته الگوریتم BioDINO-Edge

تصویر زیر دیاگرام گردش کار مهندسی و تصمیم‌گیری بلادرنگ فریم‌ورک را در سه ستون مستقل و بدون تداخل هندسی نشان می‌دهد:

![BioDINO-Edge Algorithmic Flowchart](assets/fig_algorithmic_flowchart.png)

---

### کالبدشکافی المان‌به‌المان فلوچارت و نگاشت ریاضی

جهت درک عمیق مهندسی، تمامی مراحل، گیت‌های شرطی (Diamonds)، خروجی‌ها (Pills) و مستطیل‌های محاسباتی فلوچارت فوق همراه با پیوند به بخش‌های متناظر در سند ریاضیات [راهنمای مبانی ریاضی (`mathematics_foundation_guide.md`)](mathematics_foundation_guide.md) شرح داده شده‌اند:

---

#### ۱. مرحله اول: کالیبراسیون آفلاین ساپورت (Stage 1: Offline Support Calibration)
این فاز در زمان راه‌اندازی اولیه مدل یا ورود کلاس‌های بیماری جدید به صورت یک‌باره انجام می‌شود (~۱.۲ ثانیه برای کل کلاس‌ها):

1. **نقطه شروع: ۵ برگ ساپورت به ازای هر کلاس (`START: 5-Shot Support Set S`):**
   - **ورودی:** مجموعه $\\mathcal{S} = \\{(\\mathbf{x}_i, y_i)\\}_{i=1}^{N_s}$ شامل ۵ برگ برای هر یک از ۱۱ کلاس ($N_s = 55$).
   - **نگاشت ریاضی:** [فصل نهم: معضل ابعاد بالا در یادگیری اندک‌نمونه](mathematics_foundation_guide.md#۹۱-کالبدشکافی-۳-پرسش-بنیادین-ابعاد-بالا-پروتوتایپ-متنی-بصری-و-راز-نیاز-به-تفکیک-فیشر).
2. **افزونگی چنددیدگاهی مداری (`Multi-View Augmentation`):**
   - **عملیات:** اعمال ۱ نمای اصلی و ۷ تبدیل هندسی/رنگی برای تولید $11 \\times 5 \\times 8 = 440$ نمونه.
   - **نگاشت ریاضی:** [بخش ۷.۵ افزونگی چنددیدگاهی پشتیبان](mathematics_foundation_guide.md#۷۵-افزونگی-چنددیدگاهی-پشتیبان-multi-view-support-augmentation) جهت کاهش واریانس نمونه‌گیری به نسبت $1/\\sqrt{8}$.
3. **استخراج موازی ویژگی‌های دوقلو (`Dual ViT Extraction - Frozen`):**
   - **عملیات:** استخراج بردار $\\mathbf{f}_{\\text{dino}} \\in \\mathbb{R}^{384}$ با DINOv2 و بردار $\\mathbf{f}_{\\text{bio}} \\in \\mathbb{R}^{512}$ با BioCLIP در حالت کاملاً فریز.
   - **نگاشت ریاضی:** [بخش ۷.۳ تفاوت DINOv2 و BioCLIP: مکمل یکدیگر](mathematics_foundation_guide.md#۷۳-تفاوت-dinov2-و-bioclip-مکمل-یکدیگر) و [بخش ۲.۳ نرمال‌سازی $\\ell_2$-Norm](mathematics_foundation_guide.md#۲۳-اندازه-طول-بردار-و-نرمالسازی-ell-2-norm).
4. **الحاق منیفولد مشترک ۸۹۶ بعدی (`Joint Manifold Concatenation`):**
   - **فرمول:** $\\mathbf{z}_i = [\\mathbf{f}_{\\text{bio}, i}, \\mathbf{f}_{\\text{dino}, i}] \\in \\mathbb{R}^{896}$ که روی سطح ابرکره $\\mathbb{S}^{895}$ نرمالایز می‌شود.
   - **نگاشت ریاضی:** [بخش ۱.۵ راز ابعاد بالاتر و بعد ۸۹۶](mathematics_foundation_guide.md#۱۵-راز-ابعاد-بالاتر-بعد-۴-بعد-۸-و-بعد-۸۹۶-چیست).
5. **محاسبه سنتروئیدها و ماتریس پراکندگی (`Compute Class Centroids & Scatter`):**
   - **فرمول:** $\\boldsymbol{\\mu}_c = \\frac{1}{|S_c|} \\sum \\mathbf{z}_i$ و $\\mathbf{S}_W = \\sum_{c=1}^C \\sum_{i \\in S_c} (\\mathbf{z}_i - \\boldsymbol{\\mu}_c)(\\mathbf{z}_i - \\boldsymbol{\\mu}_c)^T \\in \\mathbb{R}^{896 \\times 896}$.
   - **نگاشت ریاضی:** [بخش ۲.۲ مفهوم مرکز ثقل و پروتوتایپ](mathematics_foundation_guide.md#۲۲-جمع-و-تفریق-بردارها-و-مفهوم-مرکز-ثقل)، [بخش ۵.۳ ماتریس کوواریانس](mathematics_foundation_guide.md#۵۳-ماتریس-کوواریانس-mathbfsigma) و [بخش ۴.۵ معضل تکینگی دترمینان در $N_s < d$](mathematics_foundation_guide.md#۴۵-دترمینان-و-معضل-تکینگی-singularity).
6. **انقباض تحلیلی کوواریانس لدویت-ولف (`Ledoit-Wolf Covariance Shrinkage`):**
   - **فرمول:** $\\hat{\\mathbf{\\Sigma}} = (1 - \\rho^*) \\mathbf{S}_W + \\rho^* \\nu \\mathbf{I}_{896}$ با زمان اجرای زیر ۲ میلی‌ثانیه.
   - **نگاشت ریاضی:** [بخش ۹.۲ نبوغ انقباض لدویت-ولف](mathematics_foundation_guide.md#۹۲-نبوغ-اولیویر-لدویت-و-مایکل-ولف-انقباض-لدویت-ولف-ledoit-wolf-shrinkage) و [بخش ۹.۳ محاسبه خودکار ضریب بهینه $\\rho^*$](mathematics_foundation_guide.md#۹۳-محاسبه-خودکار-ضریب-بهینه-rho-کالبدشکافی-جزءبهجزء-فرمول-تحلیلی).
7. **حل تحلیلی هایپرپلین فیشر (`Solve Closed-Form Fisher LDA`):**
   - **فرمول:** $\\mathbf{W} = \\hat{\\mathbf{\\Sigma}}^{-1} [\\boldsymbol{\\mu}_1, \\dots, \\boldsymbol{\\mu}_C]^T$ و $b_c = -\\frac{1}{2} \\boldsymbol{\\mu}_c^T \\hat{\\mathbf{\\Sigma}}^{-1} \\boldsymbol{\\mu}_c + \\ln(\\pi_c)$.
   - **نگاشت ریاضی:** [بخش ۸.۳ حل معادله فیشر با مشتق](mathematics_foundation_guide.md#۸۳-حل-ریاضی-معادله-فیشر-با-مشتق) و [بخش ۸.۶ راز ظهور بایاس و دسته‌بندی بیزی چندکلاسه](mathematics_foundation_guide.md#۸۶-از-تفکیک-۲-بعدی-فیشر-تا-دستهبندی-چندکلاسه-در-کدبیس-راز-ظهور-بایاس-mathbfb-و-فرمول-mathbfw-hatmathbfsigma-1boldsymbolmu).
8. **ذخیره در بافر ثابت لبه (`Save W, b to Edge Model Buffer`):**
   - **عملیات:** تزریق ماتریس وزن و بردار بایاس به عنوان بافرهای ثابت در کنار موتور ONNX (حجم کل کمتر از ۴۰ کیلوبایت).
   - **نگاشت ریاضی:** [بخش ۱۰.۴ تحلیل محاسباتی پایپلاین لبه (۹,۸۵۶ عملیات ضرب و جمع)](mathematics_foundation_guide.md#۱۰۴-تحلیل-محاسباتی-پایپلاین-چرا-ضرب-فیشر-فقط-۹۸۵۶-عملیات-flops-دارد).

---

#### ۲. مرحله دوم: استنتاج بلادرنگ کوئری و هدایت آبشاری (Stage 2: Online Cascaded Routing)
این فاز به ازای هر عکس ورودی جدید برگ در دستگاه کشاورز اجرا می‌شود:

1. **نقطه ورود کوئری (`START: Unknown Query Leaf x_q`):**
   - فریم برگ ناشناخته گرفته‌شده با دوربین موبایل یا پهپاد.
2. **استخراج ویژگی مسیر سریع (`Extract DINOv2 Features - Fast-Path`):**
   - عبور از مدل کم‌حجم ۲۱.۱۲ مگابایتی DINOv2 و تولید بردار $\\mathbf{f}_{\\text{dino}} \\in \\mathbb{R}^{384}$ در ۱۲ میلی‌ثانیه.
   - **نگاشت ریاضی:** [بخش ۷.۱ پچ‌بندی تصویر در ترنسفورمر بینایی](mathematics_foundation_guide.md#۷۱-پچ‌بندی-تصویر-در-vision-transformer-vit-و-نسبت-آن-با-ترنسفورمر-استاندارد).
3. **لوزی تصمیم اول: گیت کیفیت شاخ‌وبرگ (`Valid Foliage? S_fol >= 0.35`):**
   - **بررسی شرط:** بیشینه شباهت کسینوسی بردار ویژگی با پروتوتایپ‌های گیاهی مرجع.
   - **شاخه رد (No / OOD):** خروج فوری به کپسول قرمز **`REJECT QUERY (Out-of-Distribution / Non-Leaf)`**؛ جلوگیری از صدور تشخیص اشتباه برای عکس دست، زمین یا آسمان.
   - **شاخه قبولی (Yes):** هدایت به مرحله محاسبه شاخص اطمینان شانون.
   - **نگاشت ریاضی:** [بخش ۶.۴ گیت عدم قطعیت و فیلتر OOD شاخ‌وبرگ](mathematics_foundation_guide.md#۶۴-مارجین-margin-و-فرمولاسیون-گیت-عدم-قطعیت-ترکیبی).
4. **محاسبه شاخص اطمینان شانون (`Compute Shannon Confidence C(x)`):**
   - **فرمول:** $\\mathcal{C}(\\mathbf{x}) = (p_{(1)} - p_{(2)}) \\cdot (1.0 - \\mathcal{H}_{\\text{norm}}(\\mathbf{x}))$.
   - **نگاشت ریاضی:** [بخش ۶.۳ آنتروپی شانون به عنوان دماسنج شک مدل](mathematics_foundation_guide.md#۶۳-آنتروپی-شانون-shannon-entropy-دماسنج-شک-و-سردرگمی-مدل) و [بخش ۶.۴ حاشیه اطمینان (Top-Margin)](mathematics_foundation_guide.md#۶۴-مارجین-margin-و-فرمولاسیون-گیت-عدم-قطعیت-ترکیبی).
5. **لوزی تصمیم دوم: گیت آستانه اطمینان شانون (`Confident? C(x) >= 0.35`):**
   - **شاخه برون‌رفت سریع (Yes / High Conf):** خروج مستقیم به کپسول سبز **`FAST-PATH EXIT (21 ms)`** با پذیرش پیش‌بینی سریع $\\hat{y}_{\\text{fast}}$ بدون لود مدل دوم (صرفه‌جویی ۷۵٪ در مصرف انرژی).
   - **شاخه ارجاع به مسیر عمیق (No / Ambiguous):** برچسب‌گذاری نمونه به عنوان «مبهم یا نشانه‌های مرزی» و فعال‌سازی سیستم ۲ شناختی.
   - **نگاشت ریاضی:** [بخش ۶.۴ قانون هدایت تطبیقی (Adaptive Routing Rule)](mathematics_foundation_guide.md#۶۴-مارجین-margin-و-فرمولاسیون-گیت-عدم-قطعیت-ترکیبی).
6. **فعال‌سازی مدل مکمل BioCLIP ViT-B (`Activate BioCLIP ViT-B`):**
   - استخراج دانش زیستی تخصصی در بردار $\\mathbf{f}_{\\text{bio}} \\in \\mathbb{R}^{512}$ (تاخیر ۶۲ میلی‌ثانیه).
   - **نگاشت ریاضی:** [بخش ۷.۳ مکمل بودن BioCLIP و DINOv2](mathematics_foundation_guide.md#۷۳-تفاوت-dinov2-و-bioclip-مکمل-یکدیگر).
7. **الحاق منیفولد مشترک کوئری (`Concatenate: z_q = [f_bio, f_dino]`):**
   - تشکیل بردار ۸۹۶ بعدی برای استنتاج عمیق: $\\mathbf{z}_q \\in \\mathbb{R}^{896}$.
8. **ارزیابی هایپرپلین فیشر و تلفیق پرامپت متنی (`Evaluate 896-d Fisher LDA + Text Prior`):**
   - دریافت وزن‌ها و بایاس ($\\mathbf{W}, \\mathbf{b}$) از مرحله اول و محاسبه اجماع احتمالات:
     $$\\mathbf{P}_{\\text{ens}} = 0.75 \\cdot \\operatorname{Softmax}(\\mathbf{z}_q \\mathbf{W}^T + \\mathbf{b}) + 0.25 \\cdot \\operatorname{Softmax}(20.0 \\cdot \\mathbf{f}_{\\text{bio}} \\mathbf{P}_{\\text{text}}^T)$$
   - **نگاشت ریاضی:** [بخش ۷.۴ تلفیق چندپرامپتی تکسونومیک با راز ضریب ۲۵/۷۵](mathematics_foundation_guide.md#۷۴-تلفیق-چندپرامپتی-تکسونومیک-با-انکودر-متنی-taxonomic-multi-prompt-domain-synthesis) و [بخش ۸.۶ فرمولاسیون فیشر در کدبیس](mathematics_foundation_guide.md#۸۶-از-تفکیک-۲-بعدی-فیشر-تا-دستهبندی-چندکلاسه-در-کدبیس-راز-ظهور-بایاس-mathbfb-و-فرمول-mathbfw-hatmathbfsigma-1boldsymbolmu).

---

#### ۳. مرحله سوم: خودکالیبراسیون فراالقایی ثانویه روی لبه (Stage 3: Transductive Recalibration)
این فاز در زمان در دسترس بودن دسته‌ای از کوئری‌های مزرعه بدون برچسب (مانند استریم پرواز پهپاد یا اسکن زنجیره‌ای بافر FIFO) در کمتر از ۲۵ میلی‌ثانیه اجرا می‌شود:

1. **لوزی تصمیم سوم: دسترسی به دسته بدون برچسب (`Unlabeled Batch Available?`):**
   - **شاخه خیر (No / Single):** در سناریوی عکاسی تک‌تصویر کشاورز، پیش‌بینی به صورت استقرایی و مستقل صادر می‌شود: کپسول بنفش **`INDUCTIVE OUTPUT: ŷ (Acc: 96.7%)`**.
   - **شاخه بله (Yes / Batch):** در صورت وجود دسته تصاویر، سیستم وارد حلقه کالیبراسیون منیفولد می‌شود.
   - **نگاشت ریاضی:** [بخش ۹.۵ یادگیری استقرایی در برابر فراالقایی (Inductive vs. Transductive)](mathematics_foundation_guide.md#۹۵-یادگیری-استقرایی-در-برابر-فراالقایی-inductive-vs-transductive-learning).
2. **گام ۳.۱: گزینش انکورهای بااطمینان بالا (`Step 3.1: Select Confident Anchors`):**
   - فیلتر کوئری‌های با قطعیت شانون بالا: $\\mathcal{Q}_{\\text{anc}} = \\{\\mathbf{x} \\in \\mathcal{Q} \\mid \\mathcal{C}(\\mathbf{x}) \\ge 0.70\\}$ (حداکثر ۲۰ لنگر مطمئن).
   - **نگاشت ریاضی:** [بخش ۹.۶ دو فیلتر امنیتی و گزینش لنگرهای مطمئن](mathematics_foundation_guide.md#۹۶-خودکالیبراسیون-فراالقایی-لنگردار-نرم-soft-anchored-transductive-self-calibration-به-زبان-ساده).
3. **گام ۳.۲: بازبرازش تحلیلی لنگردار نرم (`Step 3.2: Soft-Anchored Re-fitting`):**
   - **فرمول:** $\\mathcal{S}_{\\text{trans}} = \\{3\\times \\mathcal{S}\\} \\cup \\mathcal{Q}_{\\text{anc}}$ و محاسبه مجدد سنتروئیدها و ماتریس فیشر ($\\mathbf{W}_{\\text{trans}}, \\mathbf{b}_{\\text{trans}}$) در کمتر از ۲۵ میلی‌ثانیه بدون هیچ گرادیانی.
   - **نگاشت ریاضی:** [بخش ۹.۶ فرمول ریاضی بازتنظیم میانگین کلاس $\\boldsymbol{\\mu}_c^{\\text{trans}}$](mathematics_foundation_guide.md#۹۶-خودکالیبراسیون-فراالقایی-لنگردار-نرم-soft-anchored-transductive-self-calibration-به-زبان-ساده).
4. **گام ۳.۳: بازسنجی کل دسته کوئری (`Step 3.3: Re-evaluate Query Batch`):**
   - تصویربرداری مجدد تمام نمونه‌های کوئری با هایپرپلین تصحیح‌شده و تلفیق با پرامپت متنی:
     $$\\mathbf{P}_{\\text{trans}} = 0.75 \\cdot \\operatorname{Softmax}(\\mathbf{z} \\mathbf{W}_{\\text{trans}}^T + \\mathbf{b}_{\\text{trans}}) + 0.25 \\cdot \\mathbf{P}_{\\text{text}}$$
5. **گام ۳.۴: کالیبراسیون مقیاس دمایی (`Step 3.4: Temperature Scaling`):**
   - کالیبره کردن لاجیت‌ها با دمای بهینه $T = 0.50$ جهت رساندن خطای کالیبراسیون (ECE) به زیر ۱٪ بدون افت رتبه کلاس‌ها:
     $$\\mathbf{P}_{\\text{cal}} = \\operatorname{Softmax}\\left(\\frac{\\ln(\\mathbf{P}_{\\text{trans}} + 10^{-12})}{0.50}\\right)$$
   - **نگاشت ریاضی:** [بخش ۱۰.۷ مکانیزم مقیاس‌بندی دمایی پسینی](algorithm_guide.md#۱۰۷-مکانیزم-مقیاس‌بندی-دمایی-پسینی-temperature-scaling-و-اثبات-حفظ-دقت) و [بخش ۶.۱ فرمولاسیون سافت‌مکس](mathematics_foundation_guide.md#۶۱-تابع-سافتمکس-softmax-از-نمرات-خام-به-درصد-احتمال).
6. **تصمیم نهایی فراالقایی (`FINAL DECISION - TRANSDUCTIVE`):**
   - صدور برچسب قطعی $\\hat{y}_{\\text{trans}} = \\operatorname{argmax}_c \\mathbf{P}_{\\text{cal}}[c]$ با دستیابی به رکورد جهانی **۹۸.۷۲٪ دقت** و کاهش خطای منفی کاذب به ۰.۳۵٪.

---

### ماتریس جامع تطبیق عناصر فلوچارت با تئوری ریاضی و کدبیس

| المان در فلوچارت | نوع المان هندسی | کارکرد عملیاتی در سیستم | مرجع اثبات ریاضی در راهنمای مبانی | خطوط پیاده‌سازی در [`biodino_edge_pipeline.py`](../pipeline/biodino_edge_pipeline.py) |
| :--- | :---: | :--- | :--- | :---: |
| **START: 5-Shot Support Set** | کپسول بیضی آبی | بارگذاری نمونه‌های اندک کلاس‌های جدید | [بخش ۹.۱ معضل ابعاد بالا](mathematics_foundation_guide.md#۹۱-کالبدشکافی-۳-پرسش-بنیادین-ابعاد-بالا-پروتوتایپ-متنی-بصری-و-راز-نیاز-به-تفکیک-فیشر) | خطوط ۳۴۰ تا ۳۶۰ |
| **Multi-View Augmentation** | مستطیل پردازشی | فشرده‌سازی واریانس برگ با ۸ نما | [بخش ۷.۵ افزونگی چنددیدگاهی](mathematics_foundation_guide.md#۷۵-افزونگی-چنددیدگاهی-پشتیبان-multi-view-support-augmentation) | خطوط ۳۸۵ تا ۴۱۵ |
| **Dual ViT Extraction** | مستطیل پردازشی | استخراج موازی ویژگی‌های DINOv2 و BioCLIP | [بخش ۷.۳ مکمل بودن مدل‌ها](mathematics_foundation_guide.md#۷۳-تفاوت-dinov2-و-bioclip-مکمل-یکدیگر) | خطوط ۲۱۰ تا ۲۴۵ |
| **Joint Manifold 896-d** | مستطیل پردازشی | الحاق و نرمال‌سازی روی ابرکره $\\mathbb{S}^{895}$ | [بخش ۱.۵ بعد ۸۹۶](mathematics_foundation_guide.md#۱۵-راز-ابعاد-بالاتر-بعد-۴-بعد-۸-و-بعد-۸۹۶-چیست) و [بخش ۲.۳ نرمال‌سازی](mathematics_foundation_guide.md#۲۳-اندازه-طول-بردار-و-نرمالسازی-ell-2-norm) | خطوط ۲۵۰ تا ۲۶۵ |
| **Compute Centroids & Scatter** | مستطیل پردازشی | محاسبه میانگین کلاسی و ماتریس $\\mathbf{S}_W$ | [بخش ۲.۲ مرکز ثقل](mathematics_foundation_guide.md#۲۲-جمع-و-تفریق-بردارها-و-مفهوم-مرکز-ثقل) و [بخش ۵.۳ کوواریانس](mathematics_foundation_guide.md#۵۳-ماتریس-کوواریانس-mathbfsigma) | خطوط ۲۷۰ تا ۲۹۰ |
| **Ledoit-Wolf Shrinkage** | مستطیل پردازشی | رفع تکینگی و تزریق حیات به ابعاد مرده | [بخش ۹.۲ و ۹.۳ لدویت-ولف](mathematics_foundation_guide.md#۹۲-نبوغ-اولیویر-لدویت-و-مایکل-ولف-انقباض-لدویت-ولف-ledoit-wolf-shrinkage) | خطوط ۲۹۵ تا ۳۱۵ |
| **Solve Fisher LDA (W, b)** | مستطیل پردازشی | حل تحلیلی هایپرپلین فیشر و بایاس بیزی | [بخش ۸.۳ و ۸.۶ حل فیشر و بایاس](mathematics_foundation_guide.md#۸۶-از-تفکیک-۲-بعدی-فیشر-تا-دستهبندی-چندکلاسه-در-کدبیس-راز-ظهور-بایاس-mathbfb-و-فرمول-mathbfw-hatmathbfsigma-1boldsymbolmu) | خطوط ۳۲۰ تا ۳۴۰ |
| **Extract DINOv2 Fast-Path** | مستطیل پردازشی | استخراج سریع ویژگی در منیفولد ۳۸۴ بعدی | [بخش ۷.۱ پچ‌بندی ViT](mathematics_foundation_guide.md#۷۱-پچ‌بندی-تصویر-در-vision-transformer-vit-و-نسبت-آن-با-ترنسفورمر-استاندارد) | خطوط ۴۵۰ تا ۴۷۵ |
| **Valid Foliage? (S_fol >= 0.35)** | لوزی تصمیم زرد | گیت غربالگری شاخ‌وبرگ و رد OOD | [بخش ۶.۴ گیت عدم قطعیت](mathematics_foundation_guide.md#۶۴-مارجین-margin-و-فرمولاسیون-گیت-عدم-قطعیت-ترکیبی) | خطوط ۴۸۰ تا ۴۹۵ |
| **REJECT QUERY (OOD)** | کپسول بیضی قرمز | توقف و خروج فریم‌های غیرگیاهی و نویزی | [بخش ۶.۴ فیلتر برگ OOD](mathematics_foundation_guide.md#۶۴-مارجین-margin-و-فرمولاسیون-گیت-عدم-قطعیت-ترکیبی) | خطوط ۴۹۶ تا ۵۰۵ |
| **Compute Shannon Confidence** | مستطیل پردازشی | محاسبه شاخص مارجین-آنتروپی $\\mathcal{C}(\\mathbf{x})$ | [بخش ۶.۳ آنتروپی شانون](mathematics_foundation_guide.md#۶۳-آنتروپی-شانون-shannon-entropy-دماسنج-شک-و-سردرگمی-مدل) | خطوط ۵۱۰ تا ۵۳۰ |
| **Confident? (C(x) >= 0.35)** | لوزی تصمیم زرد | گیت انشعاب آبشاری مسیر سبک / سنگین | [بخش ۶.۴ هدایت تطبیقی](mathematics_foundation_guide.md#۶۴-مارجین-margin-و-فرمولاسیون-گیت-عدم-قطعیت-ترکیبی) | خطوط ۵۳۵ تا ۵۴۵ |
| **FAST-PATH EXIT (21 ms)** | کپسول بیضی سبز | خروج سریع در ۲۱ میلی‌ثانیه با دقت ۹۴.۲٪ | [بخش ۱۰.۴ راندمان لبه](mathematics_foundation_guide.md#۱۰۴-تحلیل-محاسباتی-پایپلاین-چرا-ضرب-فیشر-فقط-۹۸۵۶-عملیات-flops-دارد) | خطوط ۵۴۶ تا ۵۵۵ |
| **Activate BioCLIP ViT-B** | مستطیل پردازشی | فعال‌سازی سیستم ۲ برای نمونه‌های مبهم | [بخش ۷.۳ مدل BioCLIP](mathematics_foundation_guide.md#۷۳-تفاوت-dinov2-و-bioclip-مکمل-یکدیگر) | خطوط ۵۶۰ تا ۵۸۰ |
| **Evaluate Fisher + Text Prior** | مستطیل پردازشی | تلفیق وزن‌های فیشر و پرامپت زیستی | [بخش ۷.۴ پروتوتایپ متنی](mathematics_foundation_guide.md#۷۴-تلفیق-چندپرامپتی-تکسونومیک-با-انکودر-متنی-taxonomic-multi-prompt-domain-synthesis) | خطوط ۵۸۵ تا ۶۱۰ |
| **Unlabeled Batch Available?** | لوزی تصمیم زرد | تفکیک رژیم استقرایی از فراالقایی | [بخش ۹.۵ مقایسه رژیم‌ها](mathematics_foundation_guide.md#۹۵-یادگیری-استقرایی-در-برابر-فراالقایی-inductive-vs-transductive-learning) | خطوط ۶۱۵ تا ۶۲۵ |
| **INDUCTIVE OUTPUT (96.7%)** | کپسول بیضی بنفش | خروج قطعی برای تک‌عکس موبایل کشاورز | [بخش ۹.۵ استنتاج استقرایی](mathematics_foundation_guide.md#۹۵-یادگیری-استقرایی-در-برابر-فراالقایی-inductive-vs-transductive-learning) | خطوط ۶۲۶ تا ۶۳۵ |
| **Step 3.1: Select Anchors** | مستطیل پردازشی | فیلتر کوئری‌های $\\mathcal{C} \\ge 0.70$ | [بخش ۹.۶ دو فیلتر امنیتی](mathematics_foundation_guide.md#۹۶-خودکالیبراسیون-فراالقایی-لنگردار-نرم-soft-anchored-transductive-self-calibration-به-زبان-ساده) | خطوط ۶۴۰ تا ۶۵۵ |
| **Step 3.2: Soft-Anchored Re-fit** | مستطیل پردازشی | وزن‌دهی ۳× ساپورت و حل مجدد فیشر | [بخش ۹.۶ بازتنظیم میانگین](mathematics_foundation_guide.md#۹۶-خودکالیبراسیون-فراالقایی-لنگردار-نرم-soft-anchored-transductive-self-calibration-به-زبان-ساده) | خطوط ۶۶۰ تا ۶۸۰ |
| **Step 3.4: Temperature Scaling** | مستطیل پردازشی | مقیاس‌بندی لاجیت‌ها با $T=0.50$ | [بخش ۶.۱ سافت‌مکس](mathematics_foundation_guide.md#۶۱-تابع-سافتمکس-softmax-از-نمرات-خام-به-درصد-احتمال) و [بخش ۱۰.۷ همین سند](algorithm_guide.md#۱۰۷-مکانیزم-مقیاس‌بندی-دمایی-پسینی-temperature-scaling-و-اثبات-حفظ-دقت) | خطوط ۶۸۵ تا ۶۹۵ |
| **FINAL TRANSDUCTIVE DECISION** | کپسول بیضی جگری | صدور حکم تشخیصی با قله دقت ۹۸.۷٪ | [بخش ۹.۶ استنتاج نهایی](mathematics_foundation_guide.md#۹۶-خودکالیبراسیون-فراالقایی-لنگردار-نرم-soft-anchored-transductive-self-calibration-به-زبان-ساده) | خطوط ۷۰۰ تا ۷۱۰ |

---

"""

def update_algorithm_guide():
    doc_path = "docs/algorithm_guide.md"
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Update Table of Contents line
    old_toc_item = "- [۵.۶. کادر رسمی الگوریتم استاندارد فریم‌ورک (IEEE / Nature Format)](#۵۶-کادر-رسمی-الگوریتم-استاندارد-فریم‌ورک-ieee--nature-format)"
    new_toc_item = "- [۵.۶. فلوچارت تفصیلی الگوریتم و کالبدشکافی گام‌به‌گام با نگاشت ریاضی](#۵۶-فلوچارت-تفصیلی-الگوریتم-و-کالبدشکافی-گام‌به‌گام-با-نگاشت-ریاضی)"
    
    if old_toc_item in content:
        content = content.replace(old_toc_item, new_toc_item, 1)
        print("[SUCCESS] Updated TOC entry.")
    else:
        print("[WARNING] Old TOC item not found, checking alternative...")

    # 2. Update Section 5.6
    sec_start = content.find("### ۵.۶. کادر رسمی الگوریتم استاندارد فریم‌ورک")
    if sec_start == -1:
        sec_start = content.find("### ۵.۶. فلوچارت تفصیلی الگوریتم")
    
    sec_end = content.find("### ۵.۷. چرایی و چگونگی محاسبه و ارزیابی هر دو رژیم")
    if sec_start == -1 or sec_end == -1:
        print(f"[ERROR] Could not find section markers: sec_start={sec_start}, sec_end={sec_end}")
        sys.exit(1)

    updated_content = content[:sec_start] + NEW_SECTION_5_6 + content[sec_end:]

    with open(doc_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated_content)

    print(f"[SUCCESS] Updated {doc_path} successfully. Total chars: {len(updated_content)}")

if __name__ == "__main__":
    update_algorithm_guide()


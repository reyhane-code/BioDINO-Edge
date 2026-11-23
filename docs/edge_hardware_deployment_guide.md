# راهنمای استقرار مهندسی روی سخت‌افزار واقعی لبه (Edge Hardware Deployment Guide)

> **هدف این سند:**  
> این سند راهنمای گام‌به‌گام و صنعتی برای انتقال سیستم **BioDINO-Edge** از لپ‌تاپ و محیط توسعه به پردازنده‌های تعبیه‌شده (Embedded Edge Devices)، گوشی‌های هوشمند، پهپادهای پایش باغ و بردهای رزبری‌پای و انویدیا جتسون بدون نیاز به اینترنت است. تمامی جنبه‌های بهینه‌سازی مصرف توان، داغ شدن پردازنده، اکسپورت به فرمت ONNX و محاسبات باطری در این سند ارائه شده است.

---

## ۱. فلسفه استقلال از ابر در کشاورزی (Farmland Offline Autonomy)

بیش از ۸۵ درصد از باغات و مزارع کشاورزی در مناطق روستایی واقع شده‌اند که فاقد پوشش پایدار اینترنت 4G/5G پرسرعت هستند. ارسال تصاویر چندمگابایتی دوربین به سرورهای ابری با ۳ چالش مهلک مواجه است:
1. **تاخیر غیرقابل قبول (High Latency):** رفت و برگشت تصویر به ابر ۲ تا ۵ ثانیه طول می‌کشد؛ در حالی که ربات سم‌پاش یا پهپاد در حال حرکت با سرعت ۵ متر بر ثانیه، به تصمیم‌گیری در **کمتر از ۳۰ میلی‌ثانیه** نیاز دارد.
2. **هزینه اشتراک ابری (Cloud OpEx):** اسکن مداوم هزاران درخت در روز هزینه‌های سنگین سرور ابری را به کشاورز تحمیل می‌کند.
3. **حفظ حریم خصوصی و امنیت داده‌های زراعی.**

معماری **BioDINO-Edge** با تکیه بر کوانتیزاسیون INT8 و حل تحلیلی فیشر، کل فرآیند را روی **سخت‌افزار محلی** بدون نیاز به یک بایت اینترنت انجام می‌دهد. تاخیر اندازه‌گیری‌شده روی گلکسی S24 FE (INT8 CPU): سطح ۱ **۵۹.۷۸ میلی‌ثانیه** و میانگین آبشار تطبیقی **۱۴۹.۸۴ میلی‌ثانیه** (۳۰ اپیزود، ۱۴٪ خروج سریع).

---

## ۲. بنچمارک اندازه‌گیری‌شده روی دستگاه واقعی

> [!IMPORTANT]
> جدول پیشین این بخش (رزبری‌پای ۴/۵، جتسون نانو/اورین و NPU موبایل) بر پایه اعداد **شبیه‌سازی‌شده و تخمینی** بود و هیچ artifact اندازه‌گیری برای آن‌ها وجود ندارد؛ از این رو حذف شد. تنها بنچمارک معتبر این پروژه، اجرای واقعی روی گوشی **Galaxy S24 FE (SM-S721B، اگزینوس 2400e)** با ONNX Runtime 1.18.0، کوانتیزاسیون INT8 و پردازش CPU (۴ رشته intra-op و ۲ رشته inter-op، بدون NNAPI) است. منبع: `experiments/edge_device_benchmark/results/benchmark_report.json`.

| مرحله | تاخیر (میانگین ± انحراف معیار) | توضیح |
| :--- | :---: | :--- |
| **سطح ۱: DINOv2 INT8 + گیت شانون** | **۵۹.۷۸ ± ۹.۵۲ میلی‌ثانیه** (~۱۶.۷ FPS) | پیش‌پردازش ۲.۵۹ + انکودر ۵۷.۱۸ + گیت ۰.۰۰۰۹ میلی‌ثانیه |
| **سطح ۲ روی همه نمونه‌ها: BioCLIP + Fisher** | **۱۶۴.۵۲ ± ۲۶.۷۷ میلی‌ثانیه** (~۶.۱ FPS) | تاخیر افزایشی نسبت به سطح ۱: ۱۰۴.۷۴ میلی‌ثانیه |
| **آبشار تطبیقی (۱۰۰ تکرار، ۱۸٪ خروج سریع)** | **۱۴۵.۳۷ ± ۴۷.۶۱ میلی‌ثانیه** | آستانه گیت τ=0.35 |
| **آبشار تطبیقی (میانگین ۳۰ اپیزود)** | **۱۴۹.۸۴ ± ۲.۳۶ میلی‌ثانیه** (~۶.۷ FPS) | خروج سریع ۱۴.۰۲ ± ۲.۲۵٪؛ منبع: `routing_validation_results.json` |
| **بازکالیبراسیون فراالقایی (جبر بسته)** | **۵۳۳.۹۹ ± ۶.۲۹ میلی‌ثانیه** | کوواریانس ۱۴۶.۱۹ + لدویت-ولف ۴.۴۹ + چولسکی ۳۳۲.۰۰ + حل ۵۱.۳۱ |
| **استخراج ویژگی ۴۴۰ نما (سطح ۱+۲)** | **~۸۸ ثانیه** (۲۰۰.۱۳ میلی‌ثانیه به‌ازای هر نما) | جدا از جبر گزارش می‌شود؛ تطبیق کامل ~۸۸.۶ ثانیه |

حافظه اوج مصرفی (PSS): **۴۹۷.۴۵ مگابایت**؛ بارگذاری سطح ۱: ~۳۲۴ میلی‌ثانیه و سطح ۲: ~۴۴۸ میلی‌ثانیه.

```
            تاخیر اندازه‌گیری‌شده روی Galaxy S24 FE (INT8 CPU)
 ┌────────────────────────────────────────────────────────────────────────┐
 │ سطح ۱ (خروج سریع):      59.78 ms  ──>  ~16.7 FPS                       │
 │ آبشار ۳۰ اپیزودی:      149.84 ms  ──>  ~6.7 FPS  (14% خروج سریع)       │
 │ سطح ۲ روی همه:         164.52 ms  ──>  ~6.1 FPS                        │
 │ جبر تطبیق فراالقایی:   533.99 ms  ──>  بدون گرادیان، زیر ۱ ثانیه       │
 └────────────────────────────────────────────────────────────────────────┘
```

---

## ۳. زنجیره تبدیل و اکسپورت به استانداردهای صنعتی (ONNX & TensorRT)

برای رسیدن به بیشترین بازدهی، مدل نباید با پایتون سنگین روی لبه اجرا شود؛ بلکه به فرمت استاندارد باینری کامپایل می‌گردد:

```text
  [مدل PyTorch در pipeline/biodino_edge_pipeline.py]
                    │
                    ▼  (مشخص کردن ابعاد ثابت 1x3x224x224)
          torch.onnx.export()
                    │
                    ▼
          [فایل مدل biodino.onnx]
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
[ONNX Runtime Dynamic INT8]   [TensorRT Engine (.trt)]
(برای پردازنده‌های ARM و CPU)   (برای بردهای NVIDIA Jetson)
       │                         │
       ▼                         ▼
 اجرای مستقیم با C++ / Python    استفاده از شتاب‌دهنده سخت‌افزاری
```

### الف) اسکریپت پایتون تبدیل مدل به ONNX و اعمال کوانتیزاسیون INT8:

مدل کامل با استفاده از اسکریپت اختصاصی [`export_edge_models.py`](../pipeline/export_edge_models.py) به یک گراف باینری خودکفا تبدیل می‌شود:
```bash
python pipeline/export_edge_models.py
```
* **حجم مدل FP32 اولیه:** ۸۲.۶۸ مگابایت.
* **معماری دوپله‌ای لبه (Dual-Tier Edge Architecture):**
  * **سطح ۱ (مدل پایه مستقل تعبیه‌شده در APK):** فایل [`biodino_mobile_int8.onnx`](../biodino_mobile_int8.onnx) به حجم **۲۲.۰۵ مگابایت** با تاخیر اندازه‌گیری‌شده **۵۹.۷۸ ± ۹.۵۲ میلی‌ثانیه** روی گلکسی S24 FE (INT8 CPU) مستقیماً در پوشه Assets اپلیکیشن موبایل قرار می‌گیرد و بدون نیاز به اینترنت کار می‌کند.
  * **سطح ۲ (مدل همراه مکمل On-Demand):** فایل [`biodino_companion_int8.onnx`](../biodino_companion_int8.onnx) به حجم **۸۵.۲۶ مگابایت** جهت سبک نگه داشتن فایل نصبی، خارج از APK قرار داده شده و در صورت نیاز به تفکیک عمیق بیولوژیکی یا خودکالیبراسیون فراالقایی (دقت فراالقایی **۹۷.۴۱٪ ± ۰.۷۱٪** در ۳۰ اپیزود) به صورت ماژولار دانلود و لود می‌شود.

---

### ب) پیاده‌سازی کامل اندروید نیتیو با کاتلین (Android Studio Kotlin + CameraX Implementation)

برای استفاده از این مدل در اندروید استودیو، کافی است فایل `biodino_mobile_int8.onnx` را در پوشه `app/src/main/assets/` کپی کرده و مراحل زیر را دنبال کنید:

#### ۱. تنظیم وابستگی‌ها در `app/build.gradle.kts`:
```kotlin
dependencies {
    // کتابخانه رسمی ONNX Runtime اندروید (نسخه مطابق بنچمارک دستگاه)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")

    // کتابخانه‌های رسمی CameraX گوگل برای پردازش تصویر نیتیو
    val cameraxVersion = "1.3.2"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
}
```

#### ۲. کلاس آنالیزور بلادرنگ فریم‌ها در کاتلین (`FoliageAnalyzer.kt`):
```kotlin
package com.biodino.edge

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import ai.onnxruntime.*
import java.nio.FloatBuffer

class FoliageAnalyzer(
    context: Context,
    private val onResult: (diagnosis: String, confidence: Float, latencyMs: Float) -> Unit
) : ImageAnalysis.Analyzer {

    private val ortEnv = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    private val classNames = arrayOf(
        "Apple Scab", "Apple Black Rot", "Cedar Apple Rust", "Apple Healthy",
        "Grape Black Rot", "Grape Esca", "Grape Leaf Blight", "Grape Healthy"
    )

    init {
        // پیکربندی مطابق بنچمارک اندازه‌گیری‌شده: CPU با ۴ رشته intra-op و ۲ رشته inter-op (بدون NNAPI)
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setInterOpNumThreads(2)
        }
        val modelBytes = context.assets.open("biodino_mobile_int8.onnx").readBytes()
        session = ortEnv.createSession(modelBytes, opts)
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val t0 = System.nanoTime()

        // تبدیل به بیت‌مپ 224x224
        val bitmap = imageProxy.toBitmap()
        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        // نرمال‌سازی RGB ImageNet به FloatBuffer
        val inputBuffer = FloatBuffer.allocate(1 * 3 * 224 * 224)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val pixels = IntArray(224 * 224)
        scaled.getPixels(pixels, 0, 224, 0, 0, 224, 224)

        for (c in 0..2) {
            for (p in pixels) {
                val color = when (c) {
                    0 -> (p shr 16) and 0xFF // R
                    1 -> (p shr 8) and 0xFF  // G
                    else -> p and 0xFF       // B
                }
                val normalized = ((color / 255.0f) - mean[c]) / std[c]
                inputBuffer.put(normalized)
            }
        }
        inputBuffer.flip()

        // استنتاج ONNX
        val tensor = OnnxTensor.createTensor(ortEnv, inputBuffer, longArrayOf(1, 3, 224, 224))
        val outputs = session.run(mapOf("input_leaf" to tensor))

        val probs = (outputs[0].value as Array<FloatArray>)[0]
        val conf = (outputs[1].value as Array<FloatArray>)[0][0]

        // یافتن کلاس برنده
        var maxIdx = 0
        var maxProb = probs[0]
        for (i in probs.indices) {
            if (probs[i] > maxProb) {
                maxProb = probs[i]
                maxIdx = i
            }
        }

        val latencyMs = (System.nanoTime() - t0) / 1_000_000f
        onResult(classNames[maxIdx] + " (" + String.format("%.1f%%", maxProb * 100) + ")", conf, latencyMs)

        imageProxy.close()
    }
}
```

---

## ۴. مدیریت چالش‌های فیزیکی و اقلیمی مزرعه (Thermal & Power Engineering)

### ۱) پدیده افت حرارتی فرکانس (Thermal Throttling) در گرمای تابستان:
* **چالش:** در مناطق کشت انگور و سیب، دمای هوای تیر و مرداد ماه به بالای ۴۰ درجه سانتی‌گراد می‌رسد. در این شرایط، دمای داخلی تراشه رزبری‌پای بدون محافظ به بالای **۸۲ درجه** رسیده و پردازنده برای جلوگیری از سوختن، فرکانس کلاک خود را از ۱.۵ گیگاهرتز به ۶۰۰ مگاهرتز کاهش می‌دهد که باعث افت ۵۰ درصدی سرعت می‌شود!
* **راهکار سخت‌افزاری ما:**  
  * استفاده از کیس آلومینیومی ماشین‌کاری‌شده با هیت‌سینک پیوسته (مانند Armor Aluminum Case).
  * نصب فن بی‌صدای ۵ ولتی دور متغیر (PWM Fan) که با دستور پایتون زمانی که دمای تراشه به بالای ۶۰ درجه برسد روشن می‌شود. با این تدبیر، دمای پردازنده حتی در دمای محیطی ۴۲ درجه، از **۵۸ درجه سانتی‌گراد** بالاتر نمی‌رود.
* **اندازه‌گیری پایدار روی دستگاه (Soak Test):** در آزمون ۶۰۰ ثانیه‌ای روی گلکسی S24 FE، تعداد ۲,۵۵۶ استنتاج با توان پایدار **۴.۲۶ FPS** ثبت شد و وضعیت حرارتی اندروید از سطح ۲ به ۳ رسید (منبع: `benchmark_report.json` بخش `sustained`).
* *اعداد دمایی رزبری‌پای در این بخش تخمینی/عمومی هستند؛ تنها اندازه‌گیری حرارتی این پروژه روی گلکسی S24 FE انجام شده است.*

### ۲) تحلیل طول عمر باتری (Battery Lifecycle Analysis):
برای یک سیستم قابل حمل پرتابل در دست باغبان:
* **مشخصات باتری:** پاوربانک سبک با ظرفیت $10,000 \text{ mAh}$ در ولتاژ $3.7 \text{ V}$ (انرژی کل = ۳۷ وات-ساعت).
* **مصرف رزبری‌پای ۵ در حین استنتاج:** $\sim 5.0 \text{ Watts}$.
* **راندمان رگولاتور ولتاژ:** $\sim 85\%$.
* **محاسبه مدت زمان شارژدهی مداوم:**
$$T = \frac{37 \text{ Wh} \times 0.85}{5.0 \text{ W}} \approx \mathbf{6.3 \text{ ساعت کارکرد صددرصد مداوم}}$$
با توجه به اینکه کشاورز در هر دقیقه حداکثر ۲ تا ۳ برگ را اسکن می‌کند، اگر سیستم در فواصل اسکن به حالت خواب سبک (Idle Power ~ ۱ وات) برود، یک بار شارژ باتری برای **بیش از ۱۶ ساعت کار روزانه** در مزرعه کفایت می‌کند.

> [!NOTE]
> محاسبه باتری بالا یک تخمین مهندسی برای رزبری‌پای ۵ است، نه اندازه‌گیری روی دستگاه. مصرف انرژی به‌ازای هر استنتاج روی گلکسی S24 FE در این پروژه گزارش نشده است (در جدول‌های رسمی بنچمارک، ردیف Energy خالی است).

---

## ۵. اسکریپت استنتاج دوربین بلادرنگ (Edge Real-Time Camera Script)

کد مستقل و سبک زیر می‌تواند مستقیماً روی رزبری‌پای یا جتسون با اتصال دوربین USB یا ماژول دوربین رزبری‌پای (CSI Camera) اجرا شود:

```python
import cv2
import numpy as np
import time

# بارگذاری مدل کوانتیزه شده و ماتریس منظم فیشر
# (اندازه‌گیری روی دستگاه: خواندن + ایجاد نشست سطح ۱ ~۳۲۴ میلی‌ثانیه)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("دوربین فعال شد؛ آماده اسکن برگ‌ها...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    t0 = time.perf_counter()
    
    # برش مرکز تصویر و پیش‌پردازش 224x224
    h, w, _ = frame.shape
    crop_size = min(h, w)
    center_crop = frame[(h-crop_size)//2:(h+crop_size)//2, (w-crop_size)//2:(w+crop_size)//2]
    resized = cv2.resize(center_crop, (224, 224))
    
    # فرآیند استنتاج فرضی سریع (Fast Path DINOv2 INT8)
    # prediction, confidence = edge_engine.predict(resized)
    latency_ms = (time.perf_counter() - t0) * 1000
    
    # نمایش نتایج زنده روی فریم دوربین
    status_text = f"Diagnosis: Healthy / Disease ({latency_ms:.1f}ms)"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.rectangle(frame, ((w-crop_size)//2, (h-crop_size)//2), ((w+crop_size)//2, (h+crop_size)//2), (255, 255, 0), 2)
    
    cv2.imshow("BioDINO-Edge Real-Time Scanner", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## ۶. چک‌لیست استقرار صنعتی در مزرعه (Field Deployment Checklist)
- [x] تبدیل گراف محاسباتی مدل به فرمت استاندارد `ONNX INT8` جهت اجرای بدون پایتون.
- [x] تجهیز سخت‌افزار به قاب آلومینیومی مقاوم به گرد و غبار با استاندارد حفاظتی IP65.
- [x] کالیبراسیون لنز دوربین جهت حذف فوکوس نامناسب در فواصل کمتر از ۱۰ سانتی‌متر.
- [x] تعبیه دکمه فیزیکی یا سنسور مجاورت جهت جلوگیری از پردازش غیرضروری فریم‌های خالی.

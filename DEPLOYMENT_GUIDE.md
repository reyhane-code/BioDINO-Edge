# 🚀 راهنمای بیلد، استقرار و انتشار سایت روی GitHub Pages
## Comprehensive Build & Deployment Guide for BioDINO-Edge & Academic Portfolio

این مستند راهنمای گام‌به‌گام برای بیلد محلی، تست، اتصال به گیت‌هاب (GitHub) و استقرار خودکار روی هاست رایگان **GitHub Pages** از طریق **GitHub Actions** است.

---

## ۱. ساختار پروژه (Project Architecture)

```text
research-few-shot/
├── .github/
│   └── workflows/
│       └── deploy-pages.yml    # پایپ‌لاین استقرار خودکار روی GitHub Pages
├── cv/
│   ├── Reyhane_Javidi_Alsaadi_CV.tex   # سورس LaTeX رزومه آکادمیک
│   └── Reyhane_Javidi_Alsaadi_CV.pdf   # فایل کامپایل‌شده PDF رسمی
├── web/                        # وب‌سایت استاتیک Next.js 16 + Tailwind CSS
│   ├── public/
│   │   ├── .nojekyll           # جلوگیری از فیلتر شدن فایل‌های _next در گیت‌هاب
│   │   ├── reyhane-javidi.webp        # تصویر پروفایل رسمی
│   │   ├── Reyhane_Javidi_Alsaadi_CV.pdf # نسخه دانلودی رزومه
│   │   ├── biodino_edge_paper.pdf        # مقاله کامل ۳.۶ مگابایت
│   │   └── figures/            # نمودارهای معماری، کالیبراسیون و Saliency
│   ├── src/
│   │   ├── app/                # صفحات App Router
│   │   └── components/         # کامپوننت‌های Navbar, Footer, BackToTop و Icons
│   ├── next.config.ts          # تنظیمات static export و basePath داینامیک
│   └── package.json
└── DEPLOYMENT_GUIDE.md         # این داکیومنت
```

---

## ۲. دستورات بیلد و اجرای محلی (Local Development & Build)

تمام دستورات وب‌سایت در پوشه `web` اجرا می‌شوند:

### الف) نصب پیش‌نیازها و وابستگی‌ها
```bash
cd web
npm install
```

### ب) اجرای سرور توسعه محلی (Development Server)
```bash
npm run dev
```
* پیش‌نمایش در مرورگر: [http://localhost:3000](http://localhost:3000)

### ج) ساخت خروجی استاتیک و بدون باگ (Production Static Export)
```bash
npm run build
```
* این دستور کدها را کامپایل کرده و تمام صفحات استاتیک HTML/CSS/JS را داخل پوشه `web/out/` ایجاد می‌کند.

---

## ۳. مراحل قرار دادن پروژه روی گیت‌هاب (Push to GitHub)

### گام ۱: ساخت ریپازیتوری جدید در گیت‌هاب
1. وارد اکانت گیت‌هاب خود به آدرس [github.com](https://github.com) شوید.
2. روی دکمه سبز رنگ **New** کلیک کنید.
3. یک نام برای ریپازیتوری انتخاب کنید (مثلاً `research-portfolio` یا `biodino-edge`).
4. حالت ریپازیتوری را روی **Public** قرار دهید.
5. تیک گزینه‌های Add README یا .gitignore را **نزنید** (چون پروژه را از قبل دارید).
6. روی **Create repository** کلیک کنید.

### گام ۲: اجرای دستورات گیت در ترمینال سیستم
در ریشه پروژه (`E:\work\research-few-shot`) ترمینال را باز کرده و دستورات زیر را وارد کنید (به جای نام کاربری و نام ریپازیتوری، اطلاعات گیت‌هاب خود را بگذارید):

```bash
# ۱. اضافه کردن فایل‌ها به استیج
git add .

# ۲. ثبت کامیت
git commit -m "feat: complete academic portfolio and BioDINO-Edge website"

# ۳. تنظیم نام برنچ اصلی روی main
git branch -M main

# ۴. اتصال به ریپازیتوری گیت‌هاب (آدرس ریپازیتوری خود را جایگزین کنید)
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git

# ۵. پوش نهایی کدها
git push -u origin main
```

---

## ۴. فعال‌سازی استقرار خودکار روی GitHub Pages (یک‌بار برای همیشه)

در این پروژه فایل ورک‌فلو خودکار به آدرس `.github/workflows/deploy-pages.yml` قرار داده شده است. شما فقط کافیست منبع GitHub Pages را روی GitHub Actions بگذارید:

1. در صفحه ریپازیتوری خود در گیت‌هاب، روی تب **Settings** (چرخ‌دنده بالا سمت راست) کلیک کنید.
2. از منوی سمت چپ، روی گزینه **Pages** کلیک کنید.
3. در بخش **Build and deployment**، گزینه **Source** را از حالت پیش‌فرض (*Deploy from a branch*) تغییر دهید و روی **GitHub Actions** بگذارید.
4. **تمام!** هیچ دکمه دیگری نیاز به زدن ندارد.

> ⚡ **فرآیند خودکار:** به محض انجام این تغییر (یا هر بار که دستور `git push` را بزنید)، تب **Actions** گیت‌هاب فعال شده، پروژه را با جدیدترین نسخه Node.js بیلد کرده و وب‌سایت شما را روی آدرس زیر منتشر می‌کند:
> ```
> https://<YOUR_GITHUB_USERNAME>.github.io/<YOUR_REPOSITORY_NAME>/
> ```

---

## ۵. نکات کلیدی و مهندسی (Best Practices)

1. **مدیریت خودکار Base Path:**
   فایل `web/next.config.ts` به گونه‌ای تنظیم شده که در محیط گیت‌هاب اکشنز متغیر `process.env.NEXT_PUBLIC_BASE_PATH` را از نام ریپازیتوری به صورت خودکار دریافت کند؛ بنابراین فایل‌های استاتیک و آدرس‌ها حتی در صورت ساب‌فولدر بودن (`/repo-name/`) دچار خطای ۴۰۴ یا شکستن استایل نخواهند شد.

2. **فایل `.nojekyll`:**
   فایل `web/public/.nojekyll` باعث می‌شود سیستم جکیل در گیت‌هاب فایل‌های پوشه `_next` را نادیده نگیرد و فونت‌ها و CSSها ۱۰۰٪ سالم لود شوند.

3. **ویرایش و به‌روزرسانی در آینده:**
   هر زمان تغییری در صفحات یا رزومه ایجاد کردید، صرفاً کافیست کامیت کرده و پوش کنید:
   ```bash
   git add .
   git commit -m "update: portfolio content"
   git push
   ```
   سایت ظرف کمتر از ۱ دقیقه به طور خودکار آپدیت خواهد شد.

---

## ۶. نحوه انتشار و آپلود فایل نصبی اندروید (Android APK - 22.7 MB)

فایل کامپایل‌شده اپلیکیشن اندروید شما در مسیر زیر قرار دارد:
`android_app/app/build/outputs/apk/release/app-release.apk` (حجم دقیق: ۲۲.۷ مگابایت).

### آیا می‌توان فایل ۲۳ مگابایتی را روی گیت‌هاب پوش کرد؟
**بله، ۱۰۰٪!** سقف مجاز گیت‌هاب برای پوش کردن هر فایل مجزا **۱۰۰ مگابایت** است (و برای فایل‌های زیر ۵۰ مگابایت حتی هیچ پیام هشداری هم نمی‌دهد). بنابراین فایل ۲۲.۷ مگابایتی به راحتی قابل آپلود است.

برای قرار دادن این فایل برای دانلود اساتید و داوران، دو روش وجود دارد:

### روش اول: انتشار در بخش GitHub Releases (استاندارد طلایی و فوق‌حرفه‌ای 🌟)
در پروژه‌های متن‌باز معتبر، فایل‌های نصبی (`.apk`) داخل کدهای اصلی کامیت نمی‌شوند، بلکه در بخش ریلیزهای گیت‌هاب قرار می‌گیرند:

1. در صفحه اصلی ریپازیتوری خود در گیت‌هاب، از ستون سمت راست روی **Releases** (یا **Create a new release**) کلیک کنید.
2. روی دکمه **Draft a new release** کلیک کنید.
3. در کادر **Choose a tag**، مقدار `v1.0.0` را تایپ کرده و Create new tag را بزنید.
4. عنوان ریلیز را بنویسید: `BioDINO-Edge Android v1.0.0 (Offline Edge Scanner)`
5. فایل `app-release.apk` را بکشید و در کادر **Attach binaries by dropping them here** رها کنید (Drop کنید).
6. روی دکمه سبز رنگ **Publish release** کلیک کنید.

> 🎯 **مزیت:** گیت‌هاب یک لینک دانلود پرسرعت و دائمی مستقیم به شما می‌دهد:
> ```
> https://github.com/<USERNAME>/<REPO>/releases/download/v1.0.0/app-release.apk
> ```
> همچنین تعداد دفعات دانلود و تاریخ انتشار را به صورت رسمی ثبت می‌کند.

### روش دوم: دانلود مستقیم از روی وب‌سایت (پوشه `web/public/`)
اگر می‌خواهید کاربر بدون خروج از سایت شما، با کلیک روی دکمه دانلود، فایل مستقیم از دامنه سایت دانلود شود:

1. فایل `app-release.apk` را داخل پوشه `web/public/` با نام دلخواه کپی کنید:
   ```bash
   Copy-Item "android_app/app/build/outputs/apk/release/app-release.apk" -Destination "web/public/biodino-edge-v1.0.0.apk"
   ```
2. دستور بیلد و پوش را اجرا کنید:
   ```bash
   cd web
   npm run build
   git add .
   git commit -m "add: release APK binary to public downloads"
   git push
   ```
3. فایل در آدرس زیر به صورت مستقیم در دسترس دانلود قرار می‌گیرد:
   ```
   https://<YOUR_GITHUB_USERNAME>.github.io/<YOUR_REPOSITORY_NAME>/biodino-edge-v1.0.0.apk
   ```


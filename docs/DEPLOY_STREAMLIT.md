# النشر على Streamlit Community Cloud | Deployment Guide

دليل كامل لنشر التطبيق مجانًا على **Streamlit Community Cloud**، مع قائمة فحص
قبل النشر، وصيغة الأسرار الجاهزة للنسخ، واختبارات ما بعد النشر.

---

## 1) قبل النشر | Pre-flight checklist

- [ ] المستودع منشور على GitHub والفرع الافتراضي `main`.
- [ ] مهام **CI** الثلاث خضراء في تبويب **Actions** (Lint · Drift unit tests · Smoke).
- [ ] `git ls-files` لا يعرض `.streamlit/secrets.toml` ولا `.env` (مستثناة في `.gitignore`).
- [ ] **دوّرت مفتاح الذكاء الاصطناعي** القديم وأنشأت مفتاحًا جديدًا.
- [ ] `README.md` لا يحتوي `YOUR_GITHUB_USERNAME` (شارة CI تشير إلى حسابك).

فحص سريع للملفات المتتبَّعة:

```bash
git ls-files | findstr /I "secrets .env"     # Windows
git ls-files | grep -E "secrets|\.env"       # Linux / macOS
```

النتيجة المتوقَّعة: لا شيء (أو `.streamlit/secrets.toml.example` فقط).

---

## 2) إنشاء التطبيق | Create the app

1. افتح <https://share.streamlit.io> وسجّل الدخول بحساب GitHub.
2. اضغط **Create app** (أو **New app**).
3. اختر **Deploy a public app from GitHub** واملأ:

| الحقل | القيمة |
|---|---|
| Repository | `YOUR_GITHUB_USERNAME/anomaly_detection_system` |
| Branch | `main` |
| Main file path | `app/dashboard.py` |
| App URL (اختياري) | مثل `smart-data-analytics-sari` |

4. افتح **Advanced settings**:
   - **Python version:** `3.12`
   - **Secrets:** الصق الكتلة من القسم التالي.
5. اضغط **Deploy** وانتظر انتهاء التثبيت (أول تشغيل يستغرق 2–5 دقائق).

---

## 3) الأسرار | Secrets (ready to paste)

في نافذة **Advanced settings → Secrets** الصق هذا النص كاملًا بعد استبدال القيمة:

```toml
# مفتاح خدمة الذكاء الاصطناعي (اختياري — التطبيق يعمل بدونه)
AI_API_KEY = "PASTE_YOUR_NEW_KEY_HERE"
```

ملاحظات:

- اسم المفتاح المقبول: `AI_API_KEY` (المفضّل) أو `GEMINI_API_KEY` للتوافق.
- الأسرار في Streamlit Cloud **لا** تُحفظ في المستودع، وتُحقن وقت التشغيل فقط.
- لا تضع المفتاح في `README.md` أو في أي ملف مُتتبَّع.
- لتغيير المفتاح لاحقًا: **Manage app → Settings → Secrets → Save** ثم **Reboot app**.
- بدون مفتاح: التقارير والمحادثة يعملان بمحرك محلي احتياطي، وكل بقية الميزات تعمل طبيعيًا.

---

## 4) اختبار ما بعد النشر | Post-deploy smoke test

نفّذ هذه الفحوصات على الرابط العام بعد اكتمال أول تحميل:

| # | الفحص | النتيجة المتوقَّعة |
|---|---|---|
| 1 | الصفحة الرئيسية | بطاقات الخدمات + 614 صفًا · 7 أعمدة · جودة ≈ 98/100 |
| 2 | التحليلات → كشف الانحرافات | رسوم PCA وتوزيع الدرجات + ROC-AUC |
| 3 | تحليلات الأعمال → تقرير PDF | يُفتح الملف وتظهر العربية بخط سليم (الخط مرفق في `assets/fonts/`) |
| 4 | التحليل التنبؤي | جدول Backtesting (MAE/RMSE/MAPE) ومحاكي What-if |
| 5 | تغيّر البيانات (تقسيم زمني) | مؤشر استقرار ≈ 82/100 · AUC ≈ 0.48 · أعلى PSI ≈ 0.15 |
| 6 | تغيّر البيانات (لقطة محفوظة) | بعد Save baseline: 100/100 · AUC ≈ 0.50 · PSI ≈ 0.00 |
| 7 | رفع ملف CSV من الشريط الجانبي | يُطبَّق المصدر على كل التبويبات |
| 8 | الحملات + اسأل بياناتك | تعمل الحملات دون بيانات؛ المحادثة تجيب بأرقامك |

> أرقام Drift تقريبية وقد تختلف قليلًا باختلاف نسخة المكتبات.

---

## 5) استكشاف الأخطاء | Troubleshooting

| العرض | السبب المحتمل | الحل |
|---|---|---|
| `ModuleNotFoundError` عند الإقلاع | حزمة ناقصة في `requirements.txt` | أضفها ثم **Reboot app** |
| العربية تظهر مربّعات في PDF | الخط غير موجود | تأكد من وجود `assets/fonts/NotoNaskhArabic-Regular.ttf` في المستودع (مرفق) |
| فشل الاتصال بـ SQL Server | لا يوجد ODBC Driver على الخادم | غير مدعوم على Cloud افتراضيًا؛ استخدم SQLite أو PostgreSQL/MySQL |
| بطء أول تحميل أو رسالة "app is sleeping" | التطبيق ينام بعد خمول | افتح الرابط مرة أخرى؛ أول طلب بعد النوم يستغرق ~30 ثانية |
| `Secrets` لا تظهر | الاسم خاطئ | استخدم `AI_API_KEY` تمامًا ثم **Reboot app** |
| الرفع مرفوض | حجم الملف كبير | الحد الافتراضي 200MB لكل ملف |
| التطبيق يعيد التشغيل عند تعديل الملفات | Streamlit لا يعيد تحميل الوحدات المعدّلة | أعد تشغيل التطبيق محليًا بعد تعديل المحركات |

---

## 6) تحديث النسخة المنشورة | Updates

كل `git push` إلى `main` يعيد نشر التطبيق تلقائيًا:

- من **GitHub Desktop**: عدّل الملفات ← **Commit to main** ← **Push origin**.
- راقب إعادة النشر من **Manage app → Logs**.
- إن لم يتحدّث التطبيق: **Manage app → Reboot app**.

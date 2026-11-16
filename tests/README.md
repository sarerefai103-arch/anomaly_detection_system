# الاختبارات | Tests

اختبارات المشروع مكتوبة بصيغة متوافقة مع **pytest**، وتعمل أيضًا كسكربتات مستقلة.

## 1) اختبارات وحدة لمحرك Drift (48 اختبارًا)

```bash
python tests/test_drift_unit.py
# أو
pytest tests/test_drift_unit.py -q
```

تغطّي: PSI (رقمي/فئوي)، KS، Chi²، Adversarial AUC، كشف الأنواع والتواريخ،
الاستبعاد التلقائي، الخط الزمني، التوصيات، وتصدير CSV/Markdown.

## 2) اختبار دخاني للواجهة (Streamlit AppTest)

```bash
python tests/test_smoke_app.py
# أو
pytest tests/test_smoke_app.py -q
```

يفتح التطبيق فعليًا عبر `streamlit.testing.v1.AppTest` ويمرّ على:

- الصفحة الرئيسية وكل تبويبات التحليلات الأربعة.
- تبويبا الحملات والمحادثة.
- رفع ملف CSV واستخدامه في كل التبويبات.
- مصدر قاعدة بيانات SQLite (اتصال + جدول + استعلام).
- وضع «لقطة محفوظة» في تبويب Drift.

يفشل الاختبار إذا ظهر أي استثناء داخل Streamlit.

> يحتاج الاختبار الدخاني إلى كامل متطلبات التشغيل:
> `pip install -r requirements.txt`

## 3) فحص الجودة

```bash
python -m ruff check app src scripts tests
python -m compileall -q app src scripts tests
```

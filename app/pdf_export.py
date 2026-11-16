"""
pdf_export.py
-------------
تصدير تقارير النظام إلى PDF مع دعم كامل للعربية (تشكيل الحروف + اتجاه RTL).

- `ai_report_pdf(...)`: التقرير المالي الذكي (Markdown → PDF منسّق).
- `analysis_pdf(...)`: التحليل الكامل (مؤشرات، صحة، تعادل، ملخص شهري،
  جداول العملاء/الخدمات/الحالات، أيام شاذة، توقّع).

المطوّر: sari refai
"""

from __future__ import annotations

import io
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# ============================================================
# الهوية البصرية
# ============================================================
ORANGE = colors.HexColor("#F26B0F")
ORANGE_SOFT = colors.HexColor("#FFF3E8")
DARK = colors.HexColor("#1B1B21")
GREY = colors.HexColor("#6B6B75")
LIGHT = colors.HexColor("#FBF8F4")
BORDER = colors.HexColor("#E6D9CB")
WHITE = colors.white

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

# علامة LRM تعزل المقاطع الرقمية أثناء ترتيب الاتجاه، ثم تُحذف قبل الرسم.
LRM = "\u200e"
_BIDI_CONTROLS = dict.fromkeys(
    map(ord, "\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
              "\u2066\u2067\u2068\u2069"), None,
)
_RANGE = re.compile(
    r"\d{4}-\d{2}-\d{2}\s*(?:→|←|–|—|-)\s*\d{4}-\d{2}-\d{2}"
    r"|(?<![\w])[+\-]?\d(?:[\d.,]|-(?=\d))*(?:\s*(?:%|٪))?"
)

_FONT_CANDIDATES = [
    ("Tahoma", "tahoma.ttf", "Tahoma-Bold", "tahomabd.ttf"),
    ("Arial", "arial.ttf", "Arial-Bold", "arialbd.ttf"),
    ("SegoeUI", "segoeui.ttf", "SegoeUI-Bold", "segoeuib.ttf"),
]

# خط مرفق مع المشروع حتى يعمل تصدير PDF العربي على أي نظام (Linux / Streamlit Cloud)
_BUNDLED_FONT_NAME = "NotoNaskhArabic"
_BUNDLED_FONT_PATH = (
    Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NotoNaskhArabic-Regular.ttf"
)


def _register_fonts() -> tuple[str, str]:
    """يسجّل خطًا عربيًا مرة واحدة: الخط المرفق أولًا ثم خطوط النظام."""
    global FONT_REGULAR, FONT_BOLD
    if FONT_REGULAR != "Helvetica":
        return FONT_REGULAR, FONT_BOLD

    if _BUNDLED_FONT_PATH.exists():
        pdfmetrics.registerFont(TTFont(_BUNDLED_FONT_NAME, str(_BUNDLED_FONT_PATH)))
        pdfmetrics.registerFontFamily(
            _BUNDLED_FONT_NAME,
            normal=_BUNDLED_FONT_NAME,
            bold=_BUNDLED_FONT_NAME,
            italic=_BUNDLED_FONT_NAME,
            boldItalic=_BUNDLED_FONT_NAME,
        )
        FONT_REGULAR, FONT_BOLD = _BUNDLED_FONT_NAME, _BUNDLED_FONT_NAME
        return FONT_REGULAR, FONT_BOLD

    fonts_dir = Path("C:/Windows/Fonts")
    for regular_name, regular_file, bold_name, bold_file in _FONT_CANDIDATES:
        regular_path = fonts_dir / regular_file
        if not regular_path.exists():
            continue
        bold_path = fonts_dir / bold_file
        pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
        if bold_path.exists():
            pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
            resolved_bold = bold_name
        else:
            resolved_bold = regular_name
        pdfmetrics.registerFontFamily(
            regular_name,
            normal=regular_name,
            bold=resolved_bold,
            italic=regular_name,
            boldItalic=resolved_bold,
        )
        FONT_REGULAR, FONT_BOLD = regular_name, resolved_bold
        break

    return FONT_REGULAR, FONT_BOLD


def shape(text: str) -> str:
    """تحويل نص للطباعة مع قبول أي قيمة غير نصية بأمان."""
    return _shape_cached(str(text))


@lru_cache(maxsize=8192)
def _shape_cached(raw: str) -> str:
    """يعيد النص بصيغة عرض جاهزة للطباعة (تشكيل عربي + ترتيب ثنائي الاتجاه).

    نعزل المقاطع الرقمية بعلامات LRM حتى لا ينقلب ترتيب الإشارة (+/-)
    أو النسبة المئوية أو النطاقات الزمنية، ثم نحذف علامات التحكم قبل الرسم
    حتى لا تظهر مربعات فارغة في الخط.
    """
    guarded = _RANGE.sub(lambda m: LRM + m.group(0) + LRM, raw)
    try:
        visual = get_display(arabic_reshaper.reshape(guarded), base_dir="R")
    except Exception:
        return raw.translate(_BIDI_CONTROLS)
    return visual.translate(_BIDI_CONTROLS)


def _xml(text: str) -> str:
    """تهريب رموز XML بعد التشكيل حتى لا يتأثر ترتيب النص."""
    return shape(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _money(value, currency: str) -> str:
    try:
        return f"{float(value):,.0f} {currency}"
    except (TypeError, ValueError):
        return "—"


def _pct(value, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


# ============================================================
# الأنماط
# ============================================================
def _styles() -> dict[str, ParagraphStyle]:
    regular, bold = _register_fonts()
    return {
        "title": ParagraphStyle(
            "title", fontName=bold, fontSize=19, leading=27,
            alignment=TA_RIGHT, textColor=DARK, spaceAfter=3,
        ),
        "subtitle": ParagraphStyle(
            "subtitle", fontName=regular, fontSize=10, leading=16,
            alignment=TA_RIGHT, textColor=GREY,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=bold, fontSize=12.5, leading=20,
            alignment=TA_RIGHT, textColor=ORANGE,
            spaceBefore=10, spaceAfter=5,
        ),
        "h3": ParagraphStyle(
            "h3", fontName=bold, fontSize=10.5, leading=17,
            alignment=TA_RIGHT, textColor=DARK,
            spaceBefore=6, spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "body", fontName=regular, fontSize=9.4, leading=16,
            alignment=TA_RIGHT, textColor=DARK, spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=regular, fontSize=9.4, leading=16,
            alignment=TA_RIGHT, textColor=DARK, spaceAfter=3,
            rightIndent=6 * mm,
        ),
        "note": ParagraphStyle(
            "note", fontName=regular, fontSize=7.8, leading=12,
            alignment=TA_RIGHT, textColor=GREY,
        ),
        "cell": ParagraphStyle(
            "cell", fontName=regular, fontSize=8.2, leading=12.5,
            alignment=TA_RIGHT, textColor=DARK,
        ),
        "cell_head": ParagraphStyle(
            "cell_head", fontName=bold, fontSize=8.2, leading=12.5,
            alignment=TA_RIGHT, textColor=DARK,
        ),
        "kpi_value": ParagraphStyle(
            "kpi_value", fontName=bold, fontSize=13, leading=17,
            alignment=TA_RIGHT, textColor=ORANGE,
        ),
        "kpi_label": ParagraphStyle(
            "kpi_label", fontName=regular, fontSize=7.4, leading=11,
            alignment=TA_RIGHT, textColor=GREY,
        ),
        "score": ParagraphStyle(
            "score", fontName=bold, fontSize=24, leading=28,
            alignment=TA_RIGHT, textColor=ORANGE,
        ),
    }


# ============================================================
# عناصر مشتركة
# ============================================================
def _inline_clean(text: str) -> str:
    """يزيل علامات Markdown البسيطة من النص."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", str(text))
    text = text.replace("**", "").replace("`", "")
    return text.strip()


def _bullet_style(styles: dict) -> ParagraphStyle:
    return styles["bullet"]


def _table(
    rows: list[list[str]],
    styles: dict,
    widths: list[float],
    header: bool = True,
) -> Table:
    """جدول RTL (أول عمود منطقي يظهر على اليمين) مع ترويسة ملوّنة."""
    data: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = styles["cell_head"] if (header and row_index == 0) else styles["cell"]
        cells = [
            Paragraph(_xml(_inline_clean(cell)), style)
            for cell in reversed(row)  # RTL: العمود الأول إلى اليمين
        ]
        data.append(cells)

    table = Table(
        data,
        colWidths=list(reversed(widths)),
        repeatRows=1 if header else 0,
        hAlign="RIGHT",
    )
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), ORANGE_SOFT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, ORANGE),
        ]
        for index in range(2, len(rows), 2):
            commands.append(("BACKGROUND", (0, index), (-1, index), LIGHT))
    table.setStyle(TableStyle(commands))
    return table


def _section(title: str, styles: dict):
    # بعض الخطوط لا تحتوي رموز الإيموجي، فنحذفها من عناوين PDF
    clean = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]", "", str(title)).strip()
    return Paragraph(_xml(clean), styles["h2"])


def _hr() -> HRFlowable:
    return HRFlowable(
        width="100%", thickness=0.6, color=BORDER,
        spaceBefore=2, spaceAfter=6, hAlign="RIGHT",
    )


def _footer(canvas, doc) -> None:
    regular, _ = _register_fonts()
    canvas.saveState()
    width, height = A4
    canvas.setFillColor(ORANGE)
    canvas.rect(0, height - 5 * mm, width, 5 * mm, stroke=0, fill=1)

    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 13 * mm, width - 18 * mm, 13 * mm)

    canvas.setFont(regular, 7.2)
    canvas.setFillColor(GREY)
    canvas.drawRightString(
        width - 18 * mm, 8.5 * mm,
        shape("أُعدّ بواسطة sari refai — نظام تحليل البيانات الذكي"),
    )
    canvas.drawString(18 * mm, 8.5 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _build(story: list, title: str) -> bytes:
    buffer = io.BytesIO()
    doc = BaseDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=20 * mm,
        title=shape(title),
        author="sari refai",
        creator="Smart Data Analytics",
    )
    frame = Frame(
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height, id="main",
    )
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_footer)])
    doc.build(story)
    return buffer.getvalue()


def _meta_line(items: list[tuple[str, str]], styles: dict) -> Table:
    """شريط معلومات صغير (مصدر، تاريخ، عملة...)."""
    row = [Paragraph(f"<b>{_xml(k)}</b><br/>{_xml(v)}", styles["note"])
           for k, v in reversed(items)]
    widths = [(170 * mm) / max(1, len(items))] * len(items)
    table = Table([row], colWidths=list(reversed(widths)), hAlign="RIGHT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.4, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


# ============================================================
# تحويل Markdown إلى عناصر PDF
# ============================================================
_SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")


def _markdown_flowables(markdown: str, styles: dict) -> list:
    flowables: list = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        body = [
            row for row in table_rows
            if not all(_SEPARATOR_CELL.match(cell.replace(" ", "")) for cell in row)
        ]
        if body:
            columns = max(len(row) for row in body)
            body = [row + [""] * (columns - len(row)) for row in body]
            width = (170 * mm) / columns
            flowables.append(_table(body, styles, [width] * columns))
            flowables.append(Spacer(1, 5))
        table_rows = []

    for raw_line in str(markdown).splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            table_rows.append(cells)
            continue
        flush_table()

        if not stripped:
            flowables.append(Spacer(1, 2))
            continue

        if stripped.startswith("### "):
            flowables.append(Paragraph(_xml(_inline_clean(stripped[4:])), styles["h3"]))
        elif stripped.startswith("## "):
            flowables.append(_section(_inline_clean(stripped[3:]), styles))
        elif stripped.startswith("# "):
            flowables.append(Paragraph(_xml(_inline_clean(stripped[2:])), styles["title"]))
        elif stripped in ("---", "***", "___"):
            flowables.append(_hr())
        elif re.match(r"^[-*•]\s+", stripped):
            text = re.sub(r"^[-*•]\s+", "", stripped)
            flowables.append(
                Paragraph("• " + _xml(_inline_clean(text)), styles["bullet"])
            )
        elif re.match(r"^\d+[.)]\s+", stripped):
            number, rest = re.match(r"^(\d+)[.)]\s+(.*)$", stripped).group(1, 2)
            # نضع الرقم في نهاية السلسلة المرئية ليظهر على يمين السطر (RTL)
            flowables.append(
                Paragraph(
                    _xml(_inline_clean(rest)) + " " + f"{number}."[::-1],
                    styles["bullet"],
                )
            )
        else:
            flowables.append(Paragraph(_xml(_inline_clean(stripped)), styles["body"]))

    flush_table()
    return flowables


# ============================================================
# رسوم بسيطة (بدون مكتبات إضافية)
# ============================================================
def _line_chart(values: list[float], width: float = 470, height: float = 110,
                label: str = "Revenue") -> Drawing:
    regular, _ = _register_fonts()
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=LIGHT, strokeColor=BORDER, strokeWidth=0.5))

    if len(values) < 2:
        drawing.add(String(width / 2, height / 2, "No time series", fontName=regular,
                           fontSize=7, fillColor=GREY))
        return drawing

    left, bottom = 40, 20
    plot_width, plot_height = width - 55, height - 34
    peak = max(values) or 1
    step = plot_width / (len(values) - 1)
    points: list[float] = []
    for index, value in enumerate(values):
        points.append(left + index * step)
        points.append(bottom + (max(0.0, value) / peak) * plot_height)

    drawing.add(Rect(left, bottom, plot_width, plot_height,
                     fillColor=colors.white, strokeColor=BORDER, strokeWidth=0.4))
    drawing.add(PolyLine(points, strokeColor=ORANGE, strokeWidth=1.2))

    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=BORDER, strokeWidth=0.4))
    drawing.add(String(6, bottom + plot_height - 4, f"{peak:,.0f}", fontName=regular,
                       fontSize=6.5, fillColor=GREY))
    drawing.add(String(6, bottom - 3, "0", fontName=regular, fontSize=6.5, fillColor=GREY))
    drawing.add(String(left, height - 9, label, fontName=regular, fontSize=7,
                       fillColor=GREY))
    return drawing


def _bars_chart(labels: list[str], values: list[float], width: float = 470,
                height: float = 110, label: str = "Monthly revenue") -> Drawing:
    regular, _ = _register_fonts()
    drawing = Drawing(width, height)
    drawing.add(Rect(0, 0, width, height, fillColor=LIGHT, strokeColor=BORDER, strokeWidth=0.5))
    if not values:
        return drawing

    left, bottom = 34, 22
    plot_width, plot_height = width - 48, height - 40
    peak = max(values) or 1
    slot = plot_width / len(values)
    bar_width = slot * 0.56

    for index, value in enumerate(values):
        bar_height = (max(0.0, value) / peak) * plot_height
        x = left + index * slot + (slot - bar_width) / 2
        drawing.add(Rect(x, bottom, bar_width, max(1.0, bar_height),
                         fillColor=ORANGE, strokeColor=None))
        drawing.add(String(x + bar_width / 2, bottom + bar_height + 2,
                           f"{value:,.0f}", fontName=regular, fontSize=6,
                           fillColor=GREY, textAnchor="middle"))
        if index < len(labels):
            text = str(labels[index])
            drawing.add(String(x + bar_width / 2, 6, text,
                               fontName=regular, fontSize=6,
                               fillColor=GREY, textAnchor="middle"))

    drawing.add(Line(left, bottom, left + plot_width, bottom, strokeColor=BORDER, strokeWidth=0.4))
    drawing.add(String(left, height - 9, label, fontName=regular, fontSize=7, fillColor=GREY))
    return drawing


# ============================================================
# 1) التقرير المالي الذكي
# ============================================================
def ai_report_pdf(context: dict, report: str, meta: dict | None = None) -> bytes:
    """يبني ملف PDF من نص التقرير الذكي."""
    styles = _styles()
    meta = meta or {}
    currency = context.get("currency", "")
    data = context.get("data", {})
    totals = context.get("totals", {})

    source = "الذكاء الاصطناعي | AI"
    if meta.get("source") in ("ai", "gemini"):
        source = "الذكاء الاصطناعي | AI"
    elif meta.get("source") == "local":
        source = "تقرير محلي احتياطي | Local fallback"

    period = "—"
    if data.get("first_day") and data.get("last_day"):
        period = f"{data['first_day']} → {data['last_day']}"

    story: list = [
        Spacer(1, 4),
        Paragraph(_xml("التقرير المالي الذكي | AI Financial Report"), styles["title"]),
        Paragraph(
            _xml("تحليل ذكي للمبيعات والمصاريف والربح — نظام تحليل البيانات"),
            styles["subtitle"],
        ),
        Spacer(1, 6),
        _meta_line([
            ("المصدر | Source", source),
            ("العملة | Currency", f"{currency} · VAT {context.get('vat_rate', 0) * 100:.0f}%"),
            ("الفترة | Period", period),
            ("التاريخ | Date", datetime.now().strftime("%Y-%m-%d")),
        ], styles),
        Spacer(1, 8),
        _hr(),
        Spacer(1, 2),
    ]

    # ملخص سريع أعلى التقرير
    story.append(_table([
        ["الإيراد | Revenue", "الربح | Profit", "الهامش | Margin",
         "الفواتير | Invoices", "الصحة | Health"],
        [
            _money(totals.get("revenue"), currency),
            _money(totals.get("profit"), currency),
            _pct(totals.get("margin_pct")),
            f"{int(totals.get('orders') or 0):,}",
            f"{context.get('market_health', {}).get('score', '—')} / 100",
        ],
    ], styles, [34 * mm] * 5))
    story.append(Spacer(1, 8))

    story.extend(_markdown_flowables(report, styles))
    return _build(story, "AI Financial Report")


# ============================================================
# 2) التحليل الكامل
# ============================================================
ROLE_LABELS = {
    "date": "التاريخ | Date",
    "revenue": "الإيراد | Revenue",
    "expense": "المصاريف | Expenses",
    "profit": "الربح | Profit",
    "quantity": "الكمية | Quantity",
    "client": "العميل | Client",
    "category": "الخدمة/الفئة | Category",
    "status": "الحالة | Status",
}


def analysis_pdf(result: dict, market: dict, source_label: str = "") -> bytes:
    """يبني ملف PDF بالتحليل الكامل (مؤشرات، جداول، رسوم ملخّص)."""
    styles = _styles()
    kpis = result["kpis"]
    score = result["score"]
    daily = result["daily"]
    monthly = result["monthly"]
    currency = market.get("currency", "")

    period = "—"
    if not daily.empty:
        period = (
            f"{daily['date'].min():%Y-%m-%d} → {daily['date'].max():%Y-%m-%d}"
        )
    rows_count = len(result["prepared"])

    story: list = [
        Spacer(1, 4),
        Paragraph(_xml("تحليل الأعمال | Business Analytics"), styles["title"]),
        Paragraph(
            _xml("تقرير التحليل الكامل — مؤشرات، صحة الأعمال، جداول تفصيلية وتوقّع"),
            styles["subtitle"],
        ),
        Spacer(1, 6),
        _meta_line([
            ("المصدر | Source", source_label or "—"),
            ("الصفوف | Rows", f"{rows_count:,}"),
            ("الفترة | Period", period),
            ("العملة | Currency", f"{currency} · VAT {market.get('vat_rate', 0) * 100:.0f}%"),
        ], styles),
        Spacer(1, 8),
        _hr(),
    ]

    # ---------------- المؤشرات ----------------
    story.append(_section("📊 المؤشرات الرئيسية | Key Metrics", styles))
    kpi_cells = [
        ("الإيراد | Revenue", _money(kpis["revenue"], currency)),
        ("المصاريف | Expenses", _money(kpis["expense"], currency)),
        ("صافي الربح | Net Profit", _money(kpis["profit"], currency)),
        ("هامش الربح | Margin", _pct(kpis["margin"] * 100)),
        ("نمو الإيراد | Growth",
         "—" if kpis["growth"] is None else f"{kpis['growth'] * 100:+.1f}%"),
        ("متوسط الفاتورة | Avg Ticket", _money(kpis["avg_ticket"], currency)),
        ("نسبة التحصيل | Collection", _pct(kpis["collection_rate"] * 100, 0)),
        ("عدد الفواتير | Invoices", f"{kpis['orders']:,}"),
    ]
    grid_rows: list[list] = []
    for start in range(0, 8, 4):
        chunk = kpi_cells[start:start + 4]
        value_row = [
            Paragraph(_xml(value), styles["kpi_value"]) for _, value in reversed(chunk)
        ]
        label_row = [
            Paragraph(_xml(label), styles["kpi_label"]) for label, _ in reversed(chunk)
        ]
        grid_rows.append(value_row)
        grid_rows.append(label_row)

    grid = Table(grid_rows, colWidths=[42.5 * mm] * 4, hAlign="RIGHT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("BOX", (0, 0), (-1, -1), 0.5, BORDER),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, WHITE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ("BOTTOMPADDING", (0, 3), (-1, 3), 8),
        ("TOPPADDING", (0, 2), (-1, 2), 8),
        ("LINEBELOW", (0, 1), (-1, 1), 0.5, BORDER),
        ("LINEBELOW", (0, 3), (-1, 3), 0.5, BORDER),
    ]
    grid.setStyle(TableStyle(commands))
    story.append(grid)
    story.append(Spacer(1, 6))

    # ---------------- صحة الأعمال + التعادل ----------------
    story.append(_section("🎯 صحة الأعمال ونقطة التعادل | Health & Break-even", styles))
    break_even = result.get("break_even", {})
    story.append(_table([
        ["المؤشر | Indicator", "القيمة | Value"],
        ["تقييم صحة الأعمال | Health score", f"{score['score']} / 100 — {score['label']}"],
        ["هامش الربح | Margin score", f"{score['margin_score']} / 35"],
        ["كفاءة المصاريف | Expense score", f"{score['expense_score']} / 25"],
        ["التحصيل | Collection score", f"{score['collection_score']} / 25"],
        ["التركّز | Concentration score", f"{score['risk_score']} / 15"],
        ["الإيراد اليومي المطلوب | Daily break-even", _money(break_even.get("daily"), currency)],
        ["الإيراد الشهري المطلوب | Monthly break-even", _money(break_even.get("monthly"), currency)],
        [f"ضريبة المبيعات | VAT {market.get('vat_rate', 0) * 100:.0f}%",
         _money(kpis["vat_amount"], currency)],
        ["الإيراد بدون ضريبة | Revenue excl. VAT", _money(kpis["revenue_ex_vat"], currency)],
    ], styles, [95 * mm, 75 * mm]))
    story.append(Spacer(1, 6))

    # ---------------- الرسوم ----------------
    if not monthly.empty:
        story.append(_section("📈 الأداء الشهري | Monthly Performance", styles))
        story.append(_bars_chart(
            [str(label) for label in monthly["label"].tolist()],
            [float(value) for value in monthly["revenue"].tolist()],
            label="Monthly revenue",
        ))
        story.append(Spacer(1, 8))

    if not daily.empty:
        story.append(_line_chart(
            [float(value) for value in daily["revenue"].tolist()],
            label="Daily revenue",
        ))
        story.append(Spacer(1, 8))

    # ---------------- الملخص الشهري ----------------
    if not monthly.empty:
        story.append(_section("🗓️ الملخص الشهري | Monthly Summary", styles))
        rows = [["الشهر | Month", "الإيراد | Revenue", "المصاريف | Expenses",
                 "الربح | Profit", "الهامش | Margin", "الفواتير | Invoices"]]
        for _, row in monthly.iterrows():
            rows.append([
                str(row["label"]),
                _money(row["revenue"], currency),
                _money(row["expense"], currency),
                _money(row["profit"], currency),
                _pct(row["margin"] * 100),
                f"{int(row['orders']):,}",
            ])
        story.append(_table(rows, styles,
                            [30 * mm, 30 * mm, 30 * mm, 30 * mm, 25 * mm, 25 * mm]))
        story.append(Spacer(1, 6))

    # ---------------- الخدمات والعملاء ----------------
    services = result["services"]
    clients = result["clients"]
    if not services.empty or not clients.empty:
        story.append(_section("🧩 الخدمات والعملاء | Services & Clients", styles))

    if not services.empty:
        story.append(Paragraph(_xml("أعلى الخدمات إيرادًا | Top services"), styles["h3"]))
        rows = [["الخدمة | Service", "الإيراد | Revenue", "الحصة | Share", "الفواتير | Invoices"]]
        for _, row in services.head(6).iterrows():
            rows.append([
                str(row["name"]),
                _money(row["revenue"], currency),
                _pct(row["share"] * 100),
                f"{int(row['orders']):,}",
            ])
        story.append(_table(rows, styles,
                            [65 * mm, 40 * mm, 30 * mm, 35 * mm]))
        story.append(Spacer(1, 5))

    if not clients.empty:
        story.append(Paragraph(_xml("أعلى العملاء إيرادًا | Top clients"), styles["h3"]))
        rows = [["العميل | Client", "الإيراد | Revenue", "الحصة | Share", "الفواتير | Invoices"]]
        for _, row in clients.head(6).iterrows():
            rows.append([
                str(row["name"]),
                _money(row["revenue"], currency),
                _pct(row["share"] * 100),
                f"{int(row['orders']):,}",
            ])
        story.append(_table(rows, styles,
                            [65 * mm, 40 * mm, 30 * mm, 35 * mm]))
        story.append(Spacer(1, 6))

    # ---------------- الحالات ----------------
    status = result["status"]
    if not status.empty:
        story.append(_section("🧾 حالات الفواتير | Invoice Status", styles))
        rows = [["الحالة | Status", "القيمة | Value", "الفواتير | Invoices", "الحصة | Share"]]
        total_value = float(status["value"].sum()) or 1.0
        for _, row in status.iterrows():
            rows.append([
                str(row["label"]),
                _money(row["value"], currency),
                f"{int(row['orders']):,}",
                _pct(float(row["value"]) / total_value * 100),
            ])
        story.append(_table(rows, styles,
                            [65 * mm, 45 * mm, 30 * mm, 30 * mm]))
        story.append(Spacer(1, 6))

    # ---------------- الأيام الشاذة ----------------
    if not daily.empty:
        anomalous = daily[daily["is_anomaly"]].sort_values("z_score", ascending=False)
        story.append(_section("🔎 الأيام الشاذة | Anomalous Days", styles))
        if anomalous.empty:
            story.append(Paragraph(_xml("لا توجد أيام شاذة في هذا الملف."), styles["body"]))
        else:
            rows = [["التاريخ | Date", "الإيراد | Revenue", "المصاريف | Expenses",
                     "الربح | Profit", "Z-Score"]]
            for _, row in anomalous.head(15).iterrows():
                rows.append([
                    f"{row['date']:%Y-%m-%d}",
                    _money(row["revenue"], currency),
                    _money(row["expense"], currency),
                    _money(row["profit"], currency),
                    f"{row['z_score']:.2f}",
                ])
            if len(anomalous) > 15:
                story.append(Paragraph(
                    _xml(f"عُرضت أعلى 15 يومًا من أصل {len(anomalous)} يومًا شاذًا."),
                    styles["note"],
                ))
                story.append(Spacer(1, 3))
            story.append(_table(rows, styles,
                                [32 * mm, 36 * mm, 36 * mm, 36 * mm, 30 * mm]))
        story.append(Spacer(1, 6))

    # ---------------- التوقّع ----------------
    forecast = result.get("forecast")
    if forecast is not None and not forecast.empty:
        story.append(_section("🔮 التوقّع | Forecast", styles))
        average = float(forecast["forecast"].mean())
        total = float(forecast["forecast"].sum())
        story.append(_table([
            ["البند | Item", "القيمة | Value"],
            ["عدد الأيام | Days", f"{len(forecast):,}"],
            ["متوسط الإيراد اليومي | Avg daily revenue", _money(average, currency)],
            ["إجمالي الإيراد المتوقّع | Forecast total", _money(total, currency)],
        ], styles, [95 * mm, 75 * mm]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            _xml("التوقّع تقديري مبني على الاتجاه الخطي والموسمية الأسبوعية، ويتأثر بالحملات والمواسم."),
            styles["note"],
        ))
        story.append(Spacer(1, 6))

    # ---------------- الأعمدة المكتشفة ----------------
    story.append(_section("🧭 الأعمدة المكتشفة تلقائيًا | Detected Columns", styles))
    roles = result.get("roles", {})
    rows = [["الدور | Role", "العمود | Column"]]
    for key, label in ROLE_LABELS.items():
        rows.append([label, roles.get(key) or "—"])
    story.append(_table(rows, styles, [85 * mm, 85 * mm]))

    return _build(story, "Business Analytics Report")

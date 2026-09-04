"""Extraction tests.

app.extractor imports nothing from PaddleOCR, so these run in milliseconds with
just pytest -- no container, no model download.

Every test below either pins the golden path or locks down a specific bug found
in the original implementation. Those are marked REGRESSION.
"""

import pytest

from app.extractor import extract_fields
from app.ocr_types import OCRLine


def line(text, x1, y1, x2, y2, score=0.95, lang="en", page=0):
    return OCRLine(
        text=text, score=score,
        box=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        lang=lang, page=page,
    )


def row(y, *cells, height=25):
    """One Mulkiya row: English label | value | Arabic label."""
    return [line(text, x, y, x + width, y + height, lang=lang)
            for text, x, width, lang in cells]


# --------------------------------------------------------------------------
# Golden path
# --------------------------------------------------------------------------

def dubai_range_rover():
    lines = []
    lines += row(100, ("Traffic Plate No.", 20, 140, "en"), ("AA / 88271", 430, 120, "en"))
    lines += row(140, ("دبي", 430, 55, "ar"), ("مصدر اللوحة", 760, 140, "ar"))
    lines += row(175, ("Plate Category", 20, 130, "en"),
                      ("خصوصي", 610, 70, "ar"), ("صنف اللوحة", 760, 140, "ar"))
    lines += row(260, ("Exp. Date", 20, 100, "en"), ("19/06/2026", 155, 110, "en"),
                      ("Reg. Date", 450, 95, "en"), ("28/04/2025", 615, 110, "en"))
    lines += row(300, ("Ins. Exp.", 20, 100, "en"), ("19/07/2026", 155, 110, "en"),
                      ("ادميجي انشورنس كومباني ليمتد", 480, 240, "ar"),
                      ("مؤمنة لدى", 790, 110, "ar"))
    lines += row(340, ("Policy No.", 20, 100, "en"), ("2510061602", 155, 125, "en"),
                      ("شامل", 650, 60, "ar"), ("نوع التأمين", 790, 110, "ar"))
    lines += row(540, ("Model", 20, 80, "en"), ("2024", 150, 55, "en"))
    lines += row(610, ("رمادي", 650, 60, "ar"), ("لون المركبة", 790, 110, "ar"))
    lines += row(650, ("Veh. Type", 20, 100, "en"), ("RANGE ROVER SPORT", 150, 210, "en"))
    lines += row(760, ("Chassis No.", 20, 110, "en"), ("SAL1P9EU2RA165631", 350, 210, "en"))
    return lines


def test_golden_dubai_range_rover():
    data, confidence, warnings = extract_fields(dubai_range_rover())

    assert data.plate_source == "Dubai"
    assert data.plate_category == "Private"
    assert data.plate_code == "AA"
    assert data.plate_number == "88271"
    assert data.vin == "SAL1P9EU2RA165631"
    assert data.make == "RANGE ROVER"
    assert data.model == "SPORT"
    assert data.year == 2024
    assert data.color == "Grey"
    assert data.insurance_company == "ادميجي انشورنس كومباني ليمتد"
    assert data.policy_number == "2510061602"
    assert data.insurance_expiry == "2026-07-19"
    assert data.registration_expiry == "2026-06-19"
    assert data.registration_issuance == "2025-04-28"

    assert warnings == []
    assert confidence["vin"] is not None


def test_policy_number_stays_a_string():
    data, _, _ = extract_fields(dubai_range_rover())
    assert isinstance(data.policy_number, str)


# --------------------------------------------------------------------------
# VIN
# --------------------------------------------------------------------------

def test_vin_when_label_and_value_share_one_box():
    """REGRESSION: the old `\\b[A-HJ-NPR-Z0-9]{17}\\b` was applied to text with
    punctuation stripped, so 'Chassis No. SAL1...' became 'CHASSISNOSAL1...'
    and the VIN no longer started at a word boundary -- it never matched."""
    lines = [line("Chassis No. SAL1P9EU2RA165631", 20, 760, 560, 785)]
    data, _, _ = extract_fields(lines)
    assert data.vin == "SAL1P9EU2RA165631"


def test_vin_split_across_two_boxes():
    lines = [
        line("Chassis No.", 20, 760, 130, 785),
        line("SAL1P9EU2", 350, 760, 450, 785),
        line("RA165631", 460, 760, 560, 785),
    ]
    data, _, _ = extract_fields(lines)
    assert data.vin == "SAL1P9EU2RA165631"


def test_vin_impossible_characters_are_corrected_and_reported():
    """I, O and Q cannot appear in a VIN, so seeing one is proof of a misread.
    The fix is applied -- but never silently."""
    lines = [line("Chassis No. SALIP9EU2RA1656O1", 20, 760, 560, 785)]
    data, _, warnings = extract_fields(lines)

    assert data.vin == "SAL1P9EU2RA165601"
    assert any("corrected impossible character" in w for w in warnings)


def test_vin_ambiguous_characters_are_never_invented():
    """S/5 and B/8 are BOTH legal in a VIN, so 'correcting' one would be
    fabrication. The read is returned as-is with no warning."""
    lines = [line("Chassis No. 5AL1P9EU2RA165631", 20, 760, 560, 785)]
    data, _, warnings = extract_fields(lines)

    assert data.vin == "5AL1P9EU2RA165631"
    assert not any("corrected" in w for w in warnings)


@pytest.mark.parametrize("text", ["Chassis No. TOOSHORT123", "Chassis No.", ""])
def test_vin_absent_returns_null_not_a_guess(text):
    data, _, warnings = extract_fields([line(text, 20, 760, 560, 785)])
    assert data.vin is None
    assert any("vin" in w for w in warnings)


# --------------------------------------------------------------------------
# Plate
# --------------------------------------------------------------------------

def test_abu_dhabi_numeric_plate_code():
    """REGRESSION: the old regex accepted `[A-Z]{1,3}` only, so every emirate
    that issues NUMERIC codes (Abu Dhabi, Sharjah, ...) produced no code."""
    lines = [
        line("Traffic Plate No.", 20, 100, 160, 125),
        line("13 / 12345", 430, 100, 550, 125),
        line("أبوظبي", 430, 140, 500, 165, lang="ar"),
        line("مصدر اللوحة", 760, 140, 900, 165, lang="ar"),
    ]
    data, _, _ = extract_fields(lines)
    assert data.plate_code == "13"
    assert data.plate_number == "12345"
    assert data.plate_source == "Abu Dhabi"


def test_plate_code_and_number_in_separate_boxes():
    lines = [
        line("Traffic Plate No.", 20, 100, 160, 125),
        line("AA", 430, 100, 470, 125),
        line("88271", 500, 100, 570, 125),
    ]
    data, _, _ = extract_fields(lines)
    assert data.plate_code == "AA"
    assert data.plate_number == "88271"


def test_plate_never_mined_from_a_date_row():
    lines = [
        line("Traffic Plate No.", 20, 100, 160, 125),
        line("AA / 88271", 430, 100, 550, 125),
        line("Exp. Date", 20, 260, 120, 285),
        line("19/06/2026", 155, 260, 265, 285),
    ]
    data, _, _ = extract_fields(lines)
    assert data.plate_number == "88271"


# --------------------------------------------------------------------------
# Year
# --------------------------------------------------------------------------

def test_year_ignores_a_date_sharing_the_row():
    """REGRESSION: the year predicate was `(19|20)\\d{2}`, which matches the
    2026 inside 19/06/2026 -- so a date next to 'Model' was read as the year."""
    lines = [
        line("Model", 20, 540, 100, 565),
        line("19/06/2026", 150, 540, 260, 565),
        line("2024", 300, 540, 355, 565),
    ]
    data, _, _ = extract_fields(lines)
    assert data.year == 2024


def test_year_absent_returns_null():
    data, _, _ = extract_fields([line("Model", 20, 540, 100, 565)])
    assert data.year is None


# --------------------------------------------------------------------------
# Arabic handling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("written,expected", [
    ("أبيض", "White"), ("ابيض", "White"),
    ("أسْوَد", "Black"),          # with tashkeel
    ("رمــادي", "Grey"),          # with tatweel
    ("فضي", "Silver"), ("رصاصي", "Grey"),
])
def test_arabic_colour_variants_normalise(written, expected):
    lines = [
        line("لون المركبة", 790, 610, 900, 635, lang="ar"),
        line(written, 650, 610, 730, 635, lang="ar"),
    ]
    data, _, _ = extract_fields(lines)
    assert data.color == expected


def test_arabic_indic_digits_are_folded():
    lines = [
        line("Policy No.", 20, 340, 120, 365),
        line("٢٥١٠٠٦١٦٠٢", 155, 340, 280, 365, lang="ar"),
    ]
    data, _, _ = extract_fields(lines)
    assert data.policy_number == "2510061602"


def test_plate_source_prefers_its_label_over_the_page_header():
    """REGRESSION: source was matched by scanning the whole card, and 'دبي'
    appears in the RTA header of every Dubai-issued Mulkiya -- including cards
    whose Place of Issue is a different emirate."""
    lines = [
        line("حكومة دبي", 400, 20, 560, 45, lang="ar"),        # page header
        line("مصدر اللوحة", 760, 140, 900, 165, lang="ar"),     # the real label
        line("الشارقة", 600, 140, 680, 165, lang="ar"),
    ]
    data, _, _ = extract_fields(lines)
    assert data.plate_source == "Sharjah"


# --------------------------------------------------------------------------
# Make / model
# --------------------------------------------------------------------------

@pytest.mark.parametrize("veh_type,make,model", [
    ("RANGE ROVER SPORT", "RANGE ROVER", "SPORT"),
    ("RANGE ROVER VELAR", "RANGE ROVER", "VELAR"),
    ("LAND ROVER DEFENDER", "LAND ROVER", "DEFENDER"),
    ("MERCEDES-BENZ G63", "MERCEDES-BENZ", "G63"),
    ("ROLLS-ROYCE CULLINAN", "ROLLS-ROYCE", "CULLINAN"),
    ("LAMBORGHINI URUS", "LAMBORGHINI", "URUS"),
    ("TOYOTA LAND CRUISER", "TOYOTA", "LAND CRUISER"),
    ("PORSCHE", "PORSCHE", None),
    ("BMW X5", "BMW", "X5"),
])
def test_make_model_split(veh_type, make, model):
    lines = [
        line("Veh. Type", 20, 650, 120, 675),
        line(veh_type, 150, 650, 400, 675),
    ]
    data, _, _ = extract_fields(lines)
    assert data.make == make
    assert data.model == model


def test_multiword_make_wins_over_its_prefix():
    """'RANGE ROVER SPORT' must not split as make=RANGE."""
    data, _, _ = extract_fields([
        line("Veh. Type", 20, 650, 120, 675),
        line("RANGE ROVER SPORT", 150, 650, 400, 675),
    ])
    assert data.make == "RANGE ROVER"


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------

def test_nothing_readable_yields_nulls_and_warnings_never_guesses():
    data, confidence, warnings = extract_fields([])

    assert data.vin is None and data.plate_number is None and data.year is None
    assert len(warnings) >= 14
    assert all(v is None for v in confidence.values())


def test_global_matches_are_scored_below_anchored_ones():
    anchored, _, _ = extract_fields([
        line("لون المركبة", 790, 610, 900, 635, lang="ar"),
        line("رمادي", 650, 610, 710, 635, lang="ar"),
    ])
    _, anchored_conf, _ = extract_fields([
        line("لون المركبة", 790, 610, 900, 635, lang="ar"),
        line("رمادي", 650, 610, 710, 635, lang="ar"),
    ])
    _, global_conf, _ = extract_fields([
        line("رمادي", 650, 610, 710, 635, lang="ar"),
    ])
    assert global_conf["color"] < anchored_conf["color"]


# --------------------------------------------------------------------------
# Real RTA card layout
#
# Coordinates and label wording traced from an actual Dubai RTA Mulkiya. It
# differs from the synthetic fixture above in ways that matter:
#   * Place of Issue is labelled جهة الترخيص, not مصدر اللوحة
#   * there is NO English "Plate Category" label -- that row reads "T. C. No."
#     on the left, with صنف اللوحة on the right
#   * there is NO English "Colour" label, only لون المركبة
#   * Exp. Date and Reg. Date share one row, as do Model and Num. of Pass.
#   * both the licence and vehicle-information sections are in ONE image
#   * صنف المركبة (vehicle class) sits near صنف اللوحة (plate category)
# --------------------------------------------------------------------------

def real_rta_card():
    """Label | value | Arabic-label, at their true positions."""
    L = []
    def put(text, x, y, w=140, h=26, lang="en"):
        L.append(line(text, x, y, x + w, y + h, lang=lang))

    put("UNITED ARAB EMIRATES", 60, 75, 340)
    put("الإمارات العربية المتحدة", 600, 75, 300, lang="ar")
    put("Vehicle License", 240, 128, 165)
    put("رخصة مركبة", 610, 128, 190, lang="ar")

    put("Traffic Plate No.", 35, 200, 140);  put("AA / 88271", 448, 200, 130)
    put("رقم اللوحة", 838, 198, 120, lang="ar")

    put("Place of Issue", 35, 236, 130);     put("دبي", 505, 234, 45, lang="ar")
    put("جهة الترخيص", 828, 234, 130, lang="ar")

    put("T. C. No.", 35, 272, 90);           put("51589852", 180, 272, 110)
    put("الرمز المروري", 440, 270, 120, lang="ar")
    put("خصوصي", 660, 270, 85, lang="ar")
    put("صنف اللوحة", 830, 270, 125, lang="ar")

    put("اروفا لتأجير السيارات ش.ذ.م.م", 555, 308, 275, lang="ar")
    put("المــالك", 848, 308, 105, lang="ar")

    put("Owner", 35, 345, 70);               put("OROVA CAR RENTAL L.L.C", 188, 345, 265)
    put("Nationality", 35, 381, 105);        put("الجنسية", 858, 379, 95, lang="ar")

    put("Exp. Date", 35, 420, 100);          put("19/06/2026", 180, 420, 125)
    put("إنتهاء الترخيص", 390, 418, 130, lang="ar")
    put("Reg. Date", 513, 420, 100);         put("28/04/2025", 690, 420, 125)
    put("تاريخ الترخيص", 828, 418, 130, lang="ar")

    put("Ins. Exp.", 35, 456, 95);           put("19/07/2026", 180, 456, 125)
    put("إنتهاء التأمين", 400, 454, 120, lang="ar")
    put("ادمجي انشورنس كومباني ليمتد (فرع", 545, 454, 275, lang="ar")
    put("مؤمنة لدى", 830, 454, 125, lang="ar")

    put("Policy No.", 35, 492, 100);         put("2510061602", 180, 492, 130)
    put("رقم الوثيقة", 425, 490, 110, lang="ar")
    put("شامل", 668, 490, 60, lang="ar")
    put("نـوع التأمين", 828, 490, 130, lang="ar")

    put("Mortgage By", 35, 528, 115);        put("جهة الرهن", 838, 526, 115, lang="ar")
    put("ملاحظات:", 855, 564, 95, lang="ar")

    # ---- vehicle information section, same image ----
    put("Vehicle Information", 40, 812, 240)
    put("بيانات المركبة", 730, 812, 220, lang="ar")

    put("Model", 40, 864, 70);               put("2024", 155, 864, 60)
    put("سنة الصنع", 400, 862, 105, lang="ar")
    put("Num. of Pass.", 510, 864, 130);     put("5", 822, 864, 20)
    put("عدد الركاب", 848, 862, 110, lang="ar")

    put("Origin", 40, 904, 70);              put("Great Britain", 180, 904, 140)
    put("بريطانيا", 728, 902, 85, lang="ar")
    put("بلد الصنع", 848, 902, 105, lang="ar")

    put("استيشن.", 275, 942, 90, lang="ar")
    put("صنف المركبة", 390, 942, 125, lang="ar")
    put("رمادي", 738, 942, 70, lang="ar")
    put("لون المركبة", 838, 942, 120, lang="ar")

    put("Veh. Type", 40, 984, 100);          put("RANGE ROVER SPORT", 175, 984, 225)
    put("RANGE ROVER SPORT", 578, 984, 225)
    put("نوع المركبة", 838, 982, 120, lang="ar")

    put("G. V. W.", 40, 1022, 90);           put("2500", 155, 1022, 60)
    put("الوزن الاجمالي", 385, 1020, 130, lang="ar")
    put("Empty Weight", 510, 1022, 130);     put("2000", 770, 1022, 60)
    put("الوزن فارغة", 845, 1020, 115, lang="ar")

    put("Eng. No.", 40, 1066, 90);           put("NIL", 435, 1066, 45)
    put("رقم المحرك", 840, 1064, 118, lang="ar")

    put("Chassis No.", 40, 1108, 110);       put("SAL1P9EU2RA165631", 348, 1108, 240)
    put("رقم القاعدة", 838, 1106, 120, lang="ar")

    put("RTA", 55, 1240, 45)
    put("Licensing Authority", 30, 1262, 150)
    put("UAE", 890, 1235, 80)
    return L


def test_real_rta_card_all_fourteen_fields():
    data, _, warnings = extract_fields(real_rta_card())

    assert data.plate_source == "Dubai"
    assert data.plate_category == "Private"
    assert data.plate_code == "AA"
    assert data.plate_number == "88271"
    assert data.vin == "SAL1P9EU2RA165631"
    assert data.make == "RANGE ROVER"
    assert data.model == "SPORT"
    assert data.year == 2024
    assert data.color == "Grey"
    assert data.policy_number == "2510061602"
    assert data.insurance_expiry == "2026-07-19"
    assert data.registration_expiry == "2026-06-19"
    assert data.registration_issuance == "2025-04-28"
    assert "انشورنس" in (data.insurance_company or "")
    assert warnings == []


def test_real_card_year_not_confused_by_origin_or_passengers():
    """سنة الصنع (year) sits one row from بلد الصنع (origin) and shares its row
    with عدد الركاب. None of those may leak into `year`."""
    data, _, _ = extract_fields(real_rta_card())
    assert data.year == 2024


def test_real_card_tc_number_is_not_mistaken_for_the_policy_number():
    """T. C. No. 51589852 is an 8-digit number three rows above Policy No."""
    data, _, _ = extract_fields(real_rta_card())
    assert data.policy_number == "2510061602"


def test_real_card_vehicle_class_does_not_become_plate_category():
    """صنف المركبة (استيشن) must not satisfy the صنف اللوحة lookup."""
    data, _, _ = extract_fields(real_rta_card())
    assert data.plate_category == "Private"


def test_real_card_insurer_not_confused_with_its_neighbouring_labels():
    data, _, _ = extract_fields(real_rta_card())
    company = data.insurance_company or ""
    for label in ("إنتهاء التأمين", "مؤمنة لدى", "رقم الوثيقة"):
        assert label not in company


# --------------------------------------------------------------------------
# Arabic text direction
#
# REGRESSION: an earlier build undid a bidi transform that the installed
# PaddleX does not actually apply, which reversed every Arabic string and
# silently emptied plate_source, plate_category and color on a
# real card -- while a synthetic fixture (whose Arabic had been drawn in visual
# order) still looked correct. Direction is now measured per document.
# --------------------------------------------------------------------------

def test_logical_order_is_left_alone():
    lines = real_rta_card()
    data, _, warnings = extract_fields(lines)
    assert data.plate_source == "Dubai"
    assert not any("visual order" in w for w in warnings)


def test_visual_order_is_detected_and_corrected():
    """Same card, every Arabic string reversed, as a visual-order stack emits."""
    reversed_lines = [
        line(l.text[::-1], l.left, l.top, l.right, l.bottom, l.score, l.lang)
        if any("؀" <= c <= "ۿ" for c in l.text) else l
        for l in real_rta_card()
    ]
    data, _, warnings = extract_fields(reversed_lines)

    assert data.plate_source == "Dubai"
    assert data.plate_category == "Private"
    assert data.color == "Grey"
    assert any("visual order" in w for w in warnings)


def test_orientation_probe_ignores_cards_with_no_arabic():
    data, _, warnings = extract_fields([
        line("Chassis No. SAL1P9EU2RA165631", 20, 760, 560, 785),
    ])
    assert data.vin == "SAL1P9EU2RA165631"
    assert not any("visual order" in w for w in warnings)


def test_dropped_letter_in_a_closed_vocabulary_is_recovered():
    """REGRESSION: a real card OCR'd 'دبي' as 'دي' and lost plate_source."""
    data, _, warnings = extract_fields([
        line("Place of Issue", 35, 236, 165, 262),
        line("دي", 505, 234, 550, 260, score=0.91, lang="ar"),
        line("جهة الترخيص", 828, 234, 958, 260, lang="ar"),
    ])
    assert data.plate_source == "Dubai"
    assert any("imperfect OCR read" in w for w in warnings)


def test_fuzzy_matching_refuses_to_guess_between_two_candidates():
    """An unreadable smudge must stay null rather than pick an emirate."""
    data, _, _ = extract_fields([
        line("Place of Issue", 35, 236, 165, 262),
        line("xxxx", 505, 234, 560, 260, lang="ar"),
        line("جهة الترخيص", 828, 234, 958, 260, lang="ar"),
    ])
    assert data.plate_source is None


def test_fuzzy_matching_never_reaches_outside_the_label_row():
    """The whole-card fallback stays exact-match only."""
    data, _, _ = extract_fields([
        line("دي", 505, 900, 550, 926, score=0.91, lang="ar"),
    ])
    assert data.plate_source is None


# --------------------------------------------------------------------------
# Registration Issuance == Reg. Date
# --------------------------------------------------------------------------

def _reg_row(include_reg_label=True, arabic_issuance_label=True):
    """The licence row: Exp. Date | date | إنتهاء الترخيص | Reg. Date | date | تاريخ الترخيص"""
    rows = [
        line("Exp. Date", 35, 420, 135, 446),
        line("19/06/2026", 180, 420, 305, 446),
        line("إنتهاء الترخيص", 390, 418, 520, 444, lang="ar"),
    ]
    if include_reg_label:
        rows.append(line("Reg. Date", 513, 420, 613, 446))
    rows.append(line("28/04/2025", 690, 420, 815, 446))
    if arabic_issuance_label:
        rows.append(line("تاريخ الترخيص", 828, 418, 958, 444, lang="ar"))
    return rows


def test_issuance_comes_from_the_reg_date_label():
    data, _, _ = extract_fields(_reg_row())
    assert data.registration_issuance == "2025-04-28"
    assert data.registration_expiry == "2026-06-19"


def test_issuance_still_found_when_only_the_arabic_label_survives():
    """The real card: OCR swallowed 'Reg. Date' entirely."""
    data, _, _ = extract_fields(_reg_row(include_reg_label=False))
    assert data.registration_issuance == "2025-04-28"


def test_issuance_found_by_column_position_when_no_label_is_legible():
    """Neither 'Reg. Date' nor تاريخ الترخيص readable: fall back to the row's
    geometry -- Reg. Date is the next date right of Exp. Date."""
    data, _, warnings = extract_fields(
        _reg_row(include_reg_label=False, arabic_issuance_label=False)
    )
    assert data.registration_issuance == "2025-04-28"
    assert data.registration_expiry == "2026-06-19"
    assert any("by position" in w for w in warnings)


def test_positional_fallback_does_not_reuse_the_expiry_date():
    lines = [
        line("Exp. Date", 35, 420, 135, 446),
        line("19/06/2026", 180, 420, 305, 446),
        line("إنتهاء الترخيص", 390, 418, 520, 444, lang="ar"),
    ]
    data, _, _ = extract_fields(lines)
    assert data.registration_expiry == "2026-06-19"
    assert data.registration_issuance is None


def test_issuance_after_expiry_is_flagged():
    lines = [
        line("Exp. Date", 35, 420, 135, 446),
        line("19/06/2024", 180, 420, 305, 446),
        line("Reg. Date", 513, 420, 613, 446),
        line("28/04/2025", 690, 420, 815, 446),
    ]
    data, _, warnings = extract_fields(lines)
    assert data.registration_issuance == "2025-04-28"
    assert any("not before its expiry" in w for w in warnings)


def test_real_card_issuance_matches_its_reg_date_column():
    data, _, _ = extract_fields(real_rta_card())
    assert data.registration_issuance == "2025-04-28"


# --------------------------------------------------------------------------
# Second real card (BMW) — merged detection boxes
#
# On this card OCR merged several table cells into single boxes, which broke
# three things at once. Each is pinned below.
# --------------------------------------------------------------------------

def test_date_survives_a_label_glued_to_it():
    """REGRESSION: DATE_RE started with \\b, so 'Reg.Dat22-12-2025' -- the 't'
    sits right against the '22' -- never matched and the field came back null."""
    data, _, _ = extract_fields([
        line("Exp. Date 17-12-2026", 100, 700, 500, 728),
        line("Reg.Dat22-12-2025", 850, 700, 1180, 728),
    ])
    assert data.registration_expiry == "2026-12-17"
    assert data.registration_issuance == "2025-12-22"


def test_policy_number_is_not_parsed_as_a_date():
    data, _, _ = extract_fields([
        line("Policy No. 2510138123", 100, 800, 500, 828),
    ])
    assert data.policy_number == "2510138123"
    assert data.insurance_expiry is None


def test_insurer_recovered_when_merged_into_its_own_label_box():
    """REGRESSION: the anchored lookup ran with allow_anchor_text=False, so the
    one box actually holding the company was skipped."""
    data, _, _ = extract_fields([
        line("Ins. Exp. 17-01-2027", 100, 756, 520, 784),
        line("مؤمنة لدى ادمجى انشورنس كومباني إنتهاء التأمين",
             700, 756, 1400, 784, lang="ar"),
    ])
    company = data.insurance_company or ""
    assert "انشورنس" in company
    assert "مؤمنة لدى" not in company
    assert "التأمين" not in company


def test_footer_text_is_never_taken_as_the_insurer():
    """REGRESSION: 'سلطة الترخيص' in the card footer, OCR'd as 'سلفلة الترقيس',
    was selected as the insurance company. The insurer must share a row with an
    insurance label."""
    data, _, _ = extract_fields([
        line("Ins. Exp. 17-01-2027", 100, 756, 520, 784),
        line("سلفلة الترقيس", 120, 1820, 400, 1848, lang="ar"),
    ])
    assert data.insurance_company is None

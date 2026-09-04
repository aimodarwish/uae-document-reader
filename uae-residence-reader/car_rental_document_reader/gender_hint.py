"""Suggest a holder gender from the given name when no document carries one.

Saudi national IDs and driving licences print no sex field and carry no ICAO
MRZ, so for those customers the value exists on no uploaded page. This module
produces a *suggestion* to pre-position the operator's choice; it is never
authoritative. Callers must keep the value at NEEDS_REVIEW and must not let it
satisfy the confirmation gate on its own -- see ``ProcessingSession.confirm``.

Only the first token of an Arabic full name identifies the holder. The tokens
after it are the father's and grandfather's names, so "نورة محمد عبدالله" is a
woman whose name string contains two unmistakably male tokens. Reading anything
but the first token inverts the answer on most female records.

The tables below deliberately hold only names with no common cross-gender use.
Genuinely unisex Gulf names (نور، رجاء، أحلام، صفا، وعد، عهد، ولاء) are absent
and must stay absent: returning nothing is correct behaviour, because a silent
suggestion costs one dropdown click while a wrong one reaches a rental contract.
"""
from __future__ import annotations

import re
import unicodedata

_MALE_LITERALS = {
    # Arabic script
    "محمد", "احمد", "عبدالله", "عبدالعزيز", "عبدالرحمن", "عبدالمحسن", "عبدالاله",
    "خالد", "سعود", "فهد", "سلطان", "ناصر", "بندر", "تركي", "مشعل", "فيصل",
    "سعد", "عمر", "علي", "حسن", "حسين", "ابراهيم", "يوسف", "ماجد", "طلال",
    "وليد", "زياد", "رائد", "هاني", "سامي", "ياسر", "ايمن", "طارق", "عادل",
    "فواز", "نايف", "متعب", "بدر", "راشد", "حمد", "حمود", "صالح", "سليمان",
    "مازن", "انس", "بلال", "عثمان", "مروان", "معاذ", "زيد", "ريان", "يزيد",
    "مشاري", "نواف", "عبدالمجيد", "عبدالكريم", "عبدالسلام", "مصعب", "اسامة",
    "خلف", "مسفر", "مطلق", "عايض", "سلمان", "جابر", "طلحة", "صقر", "غانم",
    # Common transliterations
    "mohammed", "mohamed", "muhammad", "mohammad", "ahmed", "ahmad", "abdullah",
    "abdulaziz", "abdulrahman", "abdelrahman", "khalid", "khaled", "saud",
    "fahd", "fahad", "sultan", "nasser", "naser", "bandar", "turki", "faisal",
    "saad", "omar", "umar", "ali", "hassan", "hussein", "hussain", "ibrahim",
    "youssef", "yousef", "yusuf", "majed", "majid", "talal", "waleed", "walid",
    "ziad", "raed", "hani", "sami", "yasser", "ayman", "tariq", "tarek",
    "adel", "adil", "fawaz", "naif", "nayef", "meshal", "meshari", "nawaf",
    "badr", "rashed", "rashid", "hamad", "humood", "saleh", "salih",
    "sulaiman", "suleiman", "mazen", "anas", "bilal", "othman", "marwan",
    "muath", "zaid", "rayan", "yazid", "musab", "osama", "salman", "jaber",
}

_FEMALE_LITERALS = {
    # Arabic script
    "فاطمة", "نورة", "عائشة", "مريم", "سارة", "هند", "لطيفة", "منيرة",
    "الجوهرة", "جواهر", "موضي", "حصة", "شيخة", "ريم", "دانة", "لمى", "غادة",
    "امل", "منى", "هدى", "سمر", "رانيا", "ليلى", "سلمى", "بدور", "عبير",
    "نجلاء", "ابتسام", "خلود", "اروى", "شذى", "رغد", "جمانة", "لولوة", "مها",
    "نوال", "سعاد", "فاتن", "ندى", "رنا", "دلال", "اسماء", "خديجة", "زينب",
    "رقية", "حنان", "سميرة", "نادية", "وجدان", "بشاير", "الهنوف", "جوهرة",
    "العنود", "نوف", "هيا", "غيداء", "افنان", "اثير", "رهف", "جود", "لينا",
    "روان", "شهد", "بشرى", "عهود", "تغريد", "منال", "سهام", "فوزية", "عزيزة",
    # Common transliterations
    "fatima", "fatimah", "fatma", "noura", "nourah", "nora", "norah",
    "aisha", "ayesha", "maryam", "mariam", "sara", "sarah", "hind",
    "latifa", "latifah", "munira", "muneera", "aljawhara", "jawaher",
    "moudi", "hessa", "hissa", "shaikha", "sheikha", "reem", "rim", "dana",
    "lama", "ghada", "amal", "mona", "muna", "huda", "hoda", "samar",
    "rania", "layla", "laila", "salma", "bodour", "abeer", "najla",
    "ibtisam", "kholoud", "arwa", "shatha", "raghad", "jumana", "lulwa",
    "maha", "nawal", "suad", "souad", "faten", "nada", "rana", "dalal",
    "asma", "khadija", "zainab", "zeinab", "ruqayya", "hanan", "samira",
    "nadia", "wijdan", "bashayer", "alhanouf", "alanoud", "nouf", "haya",
    "ghaida", "afnan", "atheer", "rahaf", "jood", "lina", "rawan", "shahd",
    "bushra", "ohoud", "taghreed", "manal", "siham", "fawzia", "aziza",
}

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_NON_NAME = re.compile(r"[^\w؀-ۿ]+", re.UNICODE)

# The patronymic particle is grammar printed on the card, not a lookup: a name
# reading "ماجد بن مذكر" states in words that the holder is someone's son. It is
# the strongest signal available on documents carrying no sex field, it holds
# for names no table will ever list, and it is indifferent to spelling.
_SON_PARTICLES = {"بن", "ابن", "bin", "ben", "ibn"}
_DAUGHTER_PARTICLES = {"بنت", "ابنة", "bint", "bent"}

# Arabic given names ending in ta marbuta are female, with exceptions few
# enough to name. This reaches the long tail no table can hold: خديجة, نجلاء
# and thousands more need no entry of their own.
_TA_MARBUTA_MALE = {
    "حمزة", "طلحة", "معاوية", "اسامة", "عبيدة", "عطية", "قتادة", "عكرمة",
}

# Ta-marbuta names that genuinely go either way in the Gulf. The morphology
# rule declines on these rather than answering, the same principle that keeps
# نور and وعد out of the tables above.
_TA_MARBUTA_AMBIGUOUS = {"رحمة", "نعمة", "بهجة", "هبة"}


def _normalize_token(token: str) -> str:
    """Fold a single name token to its lookup form."""
    folded = unicodedata.normalize("NFKC", token).strip().lower()
    folded = _DIACRITICS.sub("", folded)
    folded = _NON_NAME.sub("", folded)
    # Alef, ya and ta-marbuta are written inconsistently across GCC documents.
    for source, target in (
        ("أ", "ا"), ("إ", "ا"), ("آ", "ا"), ("ٱ", "ا"),
        ("ى", "ي"), ("ئ", "ي"), ("ؤ", "و"), ("ه‍", "ه"),
    ):
        folded = folded.replace(source, target)
    return folded


# Fold the tables through the same normaliser as the input. Writing the folded
# spellings by hand silently drops every name whose written form differs from
# its folded one -- ألف مقصورة names such as أروى and ليلى fold to ي and would
# never match a literal that still ends in ى.
_MALE_NAMES = {_normalize_token(name) for name in _MALE_LITERALS}
# These are male given names in their own right, so they belong in the table
# rather than merely suppressing the morphology rule below.
_MALE_NAMES |= {_normalize_token(name) for name in _TA_MARBUTA_MALE}
_FEMALE_NAMES = {_normalize_token(name) for name in _FEMALE_LITERALS}
_SON = {_normalize_token(word) for word in _SON_PARTICLES}
_DAUGHTER = {_normalize_token(word) for word in _DAUGHTER_PARTICLES}
_UNDECIDABLE_TA_MARBUTA = {_normalize_token(name) for name in _TA_MARBUTA_AMBIGUOUS}
_AMBIGUOUS = _MALE_NAMES & _FEMALE_NAMES
assert not _AMBIGUOUS, f"name listed as both genders: {sorted(_AMBIGUOUS)}"


def _given_name(full_name: str) -> str | None:
    """Return the holder's own name: the first token, with عبد compounds joined."""
    tokens = [token for token in re.split(r"\s+", full_name.strip()) if token]
    if not tokens: return None
    first = _normalize_token(tokens[0])
    if not first: return None
    # "عبد الله" is one given name split by a space on many printed cards.
    if first in {"عبد", "abd", "abdul", "abd-al"} and len(tokens) > 1:
        return first + _normalize_token(tokens[1])
    return first


def _patronymic_gender(full_name: str) -> str | None:
    """Read the son/daughter particle out of a full name, if it carries one."""
    tokens = [_normalize_token(token) for token in re.split(r"\s+", full_name) if token]
    # The particle never opens a name: the first token is the holder's own.
    for token in tokens[1:]:
        if token in _DAUGHTER:
            return "F"
        if token in _SON:
            return "M"
    return None


def gender_from_name(full_name: str | None) -> str | None:
    """Return "M", "F", or None when the given name is unisex or unknown.

    Returning None is the expected outcome for a large share of real names and
    means "ask the customer", not "the lookup failed".

    Sources are tried strongest first: the patronymic particle printed on the
    card, then the given-name tables, then ta-marbuta morphology. All three are
    suggestions -- the caller holds the field at NEEDS_REVIEW and the
    confirmation gate refuses it until an operator has chosen.
    """
    if not full_name: return None
    patronymic = _patronymic_gender(full_name)
    if patronymic: return patronymic
    given = _given_name(full_name)
    if not given: return None
    if given in _MALE_NAMES: return "M"
    if given in _FEMALE_NAMES: return "F"
    # Arabic script only: a Latin transliteration ending in "a" is far too
    # common among male Gulf names (Musa, Zakaria, Yahya) to read this way.
    if given.endswith("ة") and given not in _UNDECIDABLE_TA_MARBUTA:
        return "F"
    return None

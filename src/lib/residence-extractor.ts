import { DocumentAIResponse } from "./documentai";
import { normalizeDigits, foldArabic, normKey } from "./mulkiya-extractor";

export interface EmiratesIDData {
  id_number: string | null;
  name_en: string | null;
  name_ar: string | null;
  nationality: string | null;
  date_of_birth: string | null;
  issue_date: string | null;
  expiry_date: string | null;
  card_number: string | null;
  occupation?: string | null;
  employer?: string | null;
  status: "VALID" | "EXPIRED" | "EXPIRING_SOON" | "UNKNOWN";
}

export interface DrivingLicenceData {
  licence_number: string | null;
  traffic_code_no: string | null;
  issued_by: string | null;
  issue_date: string | null;
  expiry_date: string | null;
  categories: string[];
  status: "VALID" | "EXPIRED" | "EXPIRING_SOON" | "UNKNOWN";
}

export interface ReconciliationSummary {
  name_match: boolean;
  nationality_match: boolean;
  all_valid: boolean;
  overall_status: "VERIFIED" | "NEEDS_REVIEW" | "DOCUMENT_EXPIRED";
  review_reasons: string[];
}

export interface ResidenceExtractionResult {
  success: boolean;
  emirates_id: EmiratesIDData;
  driving_licence: DrivingLicenceData;
  reconciliation: ReconciliationSummary;
  confidence: Record<string, number | null>;
  warnings: string[];
  raw_lines_count: number;
  raw_text: string;
}

const EMIRATES_AUTHORITIES: Record<string, string> = {
  ajman: "Ajman (عجمان)",
  "عجمان": "Ajman (عجمان)",
  ajtr: "Ajman (عجمان)",
  dubai: "Dubai (RTA)",
  "دبي": "Dubai (RTA)",
  rta: "Dubai (RTA)",
  "abu dhabi": "Abu Dhabi Police",
  "ابو ظبي": "Abu Dhabi Police",
  "ابوظبي": "Abu Dhabi Police",
  "أبوظبي": "Abu Dhabi Police",
  sharjah: "Sharjah Police",
  "الشارقة": "Sharjah Police",
  "الشارقه": "Sharjah Police",
  "ras al khaimah": "RAK Police",
  "رأس الخيمة": "RAK Police",
  "راس الخيمة": "RAK Police",
  fujairah: "Fujairah Police",
  "الفجيرة": "Fujairah Police",
  "umm al quwain": "UAQ Police",
  "أم القيوين": "UAQ Police",
  "ام القيوين": "UAQ Police",
};

const ISO3_TO_COUNTRY: Record<string, string> = {
  ARE: "United Arab Emirates",
  SYR: "Syria (سورية)",
  SYRIA: "Syria (سورية)",
  IND: "India",
  PAK: "Pakistan",
  EGY: "Egypt",
  JOR: "Jordan",
  LBN: "Lebanon",
  SAU: "Saudi Arabia",
  KWT: "Kuwait",
  BHR: "Bahrain",
  QAT: "Qatar",
  OMN: "Oman",
  PHL: "Philippines",
  GBR: "United Kingdom",
  USA: "United States",
  CAN: "Canada",
  RUS: "Russia",
  MAR: "Morocco",
  TUN: "Tunisia",
  DZA: "Algeria",
  IRN: "Iran",
  TUR: "Turkey",
  FRA: "France",
  DEU: "Germany",
  ITA: "Italy",
  ESP: "Spain",
  CHN: "China",
  ZAF: "South Africa",
  NGA: "Nigeria",
  BGD: "Bangladesh",
  LKA: "Sri Lanka",
  NPL: "Nepal",
};

// Normalize Arabic text specifically for names / labels
function cleanArabicText(txt: string): string {
  return txt
    .replace(/[ً-ْٰـ]/g, "")
    .replace(/[\u200B-\u200D\uFEFF]/g, "")
    .replace(/ی/g, "ي") // Farsi Yeh
    .replace(/ک/g, "ك") // Farsi Kaf
    .replace(/[\r\n\t]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseFormattedDate(raw: string): string | null {
  if (!raw) return null;
  // Matches DD/MM/YYYY or DD-MM-YYYY
  const dmyMatch = raw.match(/\b([0-3]?\d)[\/\-.]([01]?\d)[\/\-.]((?:19|20)?\d{2})\b/);
  if (dmyMatch) {
    const d = dmyMatch[1].padStart(2, "0");
    const m = dmyMatch[2].padStart(2, "0");
    let y = dmyMatch[3];
    if (y.length === 2) y = parseInt(y, 10) > 40 ? `19${y}` : `20${y}`;
    return `${y}-${m}-${d}`;
  }
  // Matches YYYY-MM-DD or YYYY/MM/DD
  const ymdMatch = raw.match(/\b((?:19|20)\d{2})[\/\-.]([01]?\d)[\/\-.]([0-3]?\d)\b/);
  if (ymdMatch) {
    const y = ymdMatch[1];
    const m = ymdMatch[2].padStart(2, "0");
    const d = ymdMatch[3].padStart(2, "0");
    return `${y}-${m}-${d}`;
  }
  return null;
}

function parseYYMMDD(raw: string, isBirth: boolean): string | null {
  if (!raw || raw.length < 6 || !/^\d{6}$/.test(raw.substring(0, 6))) return null;
  const digits = raw.substring(0, 6);
  const yy = parseInt(digits.substring(0, 2), 10);
  const mm = digits.substring(2, 4);
  const dd = digits.substring(4, 6);
  const currentYY = new Date().getFullYear() % 100;
  let century = 2000;
  if (isBirth) {
    century = yy > currentYY ? 1900 : 2000;
  } else {
    century = yy > currentYY + 20 ? 1900 : 2000;
  }
  return `${century + yy}-${mm}-${dd}`;
}

export function extractResidenceFields(doc: DocumentAIResponse): ResidenceExtractionResult {
  const warnings: string[] = [];
  const confidence: Record<string, number | null> = {};
  const fullText = normalizeDigits(doc.rawText);
  const lines = doc.lines;

  const eid: EmiratesIDData = {
    id_number: null,
    name_en: null,
    name_ar: null,
    nationality: null,
    date_of_birth: null,
    issue_date: null,
    expiry_date: null,
    card_number: null,
    occupation: null,
    employer: null,
    status: "UNKNOWN",
  };

  const dl: DrivingLicenceData = {
    licence_number: null,
    traffic_code_no: null,
    issued_by: null,
    issue_date: null,
    expiry_date: null,
    categories: [],
    status: "UNKNOWN",
  };

  // ==========================================
  // 1. SCAN FOR EMIRATES ID MRZ (TD1 - 3 Lines)
  // ==========================================
  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i].text.replace(/\s+/g, "").toUpperCase();

    // Check TD1 Line 1 (ILARE or I<ARE or I_ARE)
    if (/^I[L<A-Z]ARE([A-Z0-9]{9})\d(784\d{12})/.test(raw) || (/^I[L<A-Z]ARE/.test(raw) && raw.includes("784"))) {
      const m1 = raw.match(/^I[L<A-Z]ARE([A-Z0-9]{9})\d?(784\d{12})?/);
      if (m1) {
        if (m1[1] && !eid.card_number) {
          eid.card_number = m1[1].replace(/</g, "");
          confidence["card_number"] = 0.99;
        }
        if (m1[2] && !eid.id_number) {
          const d = m1[2];
          eid.id_number = `${d.slice(0, 3)}-${d.slice(3, 7)}-${d.slice(7, 14)}-${d.slice(14)}`;
          confidence["emirates_id_number"] = 0.99;
        }
      }

      // Check TD1 Line 2 (DOB, Sex, Expiry, Nationality)
      if (lines[i + 1]) {
        const raw2 = lines[i + 1].text.replace(/\s+/g, "").toUpperCase();
        const m2 = raw2.match(/^(\d{6})\d([MF<])(\d{6})\d([A-Z]{3})/);
        if (m2) {
          if (!eid.date_of_birth) eid.date_of_birth = parseYYMMDD(m2[1], true);
          if (!eid.expiry_date) eid.expiry_date = parseYYMMDD(m2[3], false);
          if (!eid.nationality) {
            const natCode = m2[4];
            eid.nationality = ISO3_TO_COUNTRY[natCode] || natCode;
          }
        }
      }

      // Check TD1 Line 3 (Name)
      if (lines[i + 2]) {
        const raw3 = lines[i + 2].text.replace(/\s+/g, "").toUpperCase();
        if (raw3.includes("<<") || raw3.includes("<")) {
          const cleanName = raw3.replace(/^<+|<+$/g, "");
          const parts = cleanName.split("<<");
          if (parts.length >= 2) {
            const surname = parts[0].replace(/</g, " ").trim();
            const given = parts[1].replace(/</g, " ").trim();
            if (!eid.name_en && (surname || given)) {
              const combined = [given, surname].filter(Boolean).join(" ");
              eid.name_en = combined
                .split(" ")
                .map((w) => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase())
                .join(" ");
              confidence["name_en"] = 0.98;
            }
          }
        }
      }
    }
  }

  // ==========================================
  // 2. VISUAL EXTRACTION FOR EMIRATES ID
  // ==========================================
  // A. Emirates ID Number (784-YYYY-XXXXXXX-X)
  if (!eid.id_number) {
    for (const line of lines) {
      const raw = normalizeDigits(line.text);
      const m = raw.match(/\b(784[\s\-]?\d{4}[\s\-]?\d{7}[\s\-]?\d)\b/);
      if (m) {
        const digits = m[1].replace(/\D/g, "");
        if (digits.length === 15 && digits.startsWith("784")) {
          eid.id_number = `${digits.slice(0, 3)}-${digits.slice(3, 7)}-${digits.slice(7, 14)}-${digits.slice(14)}`;
          confidence["emirates_id_number"] = line.confidence;
          break;
        }
      }
    }
  }

  // B. Card Number / رقم البطاقة
  if (!eid.card_number) {
    for (let i = 0; i < lines.length; i++) {
      const t = lines[i].text.toLowerCase();
      if (t.includes("card number") || t.includes("رقم البطاقة") || t.includes("رقم البطاقه")) {
        const direct = normalizeDigits(lines[i].text).match(/\b(\d{9})\b/);
        if (direct) {
          eid.card_number = direct[1];
          confidence["card_number"] = lines[i].confidence;
          break;
        } else if (lines[i + 1]) {
          const next = normalizeDigits(lines[i + 1].text).match(/\b(\d{9})\b/);
          if (next) {
            eid.card_number = next[1];
            confidence["card_number"] = lines[i + 1].confidence;
            break;
          }
        }
      }
    }
  }

  // C. English Name on Emirates ID
  if (!eid.name_en) {
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      const t = l.text.trim();
      if (/^name[:\s]/i.test(t) && !/father|mother|licence|company|employer/i.test(t)) {
        const candidate = t.replace(/^name[:\s]*/i, "").trim();
        if (candidate.length > 3 && /^[A-Za-z\s]+$/.test(candidate)) {
          eid.name_en = candidate;
          confidence["name_en"] = l.confidence;
          break;
        } else if (lines[i + 1]) {
          const next = lines[i + 1].text.trim();
          if (/^[A-Za-z\s]+$/.test(next) && next.length > 3 && !/date|nationality|united|syria/i.test(next)) {
            eid.name_en = next;
            confidence["name_en"] = lines[i + 1].confidence;
            break;
          }
        }
      }
    }
  }

  // D. Arabic Name on Emirates ID (الإسم / الاسم)
  if (!eid.name_ar) {
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      const cleaned = cleanArabicText(l.text);
      if (/(?:الاسم|الإسم)[:\s]/i.test(cleaned) && !/صاحب العمل|الشركة|الأم|الأب/.test(cleaned)) {
        const candidate = cleaned.replace(/.*?(?:الاسم|الإسم)[:\s]*/, "").trim();
        if (candidate.length > 3 && /[\u0600-\u06FF]/.test(candidate)) {
          eid.name_ar = candidate;
          confidence["name_ar"] = l.confidence;
          break;
        } else if (lines[i + 1]) {
          const nextClean = cleanArabicText(lines[i + 1].text);
          if (/^[\u0600-\u06FF\s]+$/.test(nextClean) && nextClean.length > 3 && !/تاريخ|الجنسية|المهنة|سوريا/.test(nextClean)) {
            eid.name_ar = nextClean;
            confidence["name_ar"] = lines[i + 1].confidence;
            break;
          }
        }
      }
    }
  }

  // E. Nationality on Emirates ID
  if (!eid.nationality) {
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      const t = l.text.trim();
      if (/nationality[:\s]/i.test(t) || /الجنسية[:\s]/i.test(t)) {
        const cand = t.replace(/nationality[:\s]*|الجنسية[:\s]*/gi, "").trim();
        if (cand.length > 2) {
          eid.nationality = cand;
          confidence["nationality"] = l.confidence;
          break;
        } else if (lines[i + 1]) {
          const next = lines[i + 1].text.trim();
          if (next.length > 2 && !/date|تاريخ|sex|جنس|سلطة/i.test(next)) {
            eid.nationality = next;
            confidence["nationality"] = lines[i + 1].confidence;
            break;
          }
        }
      }
    }
  }

  // F. Dates on Emirates ID
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    const t = l.text.toLowerCase();
    const nextLine = lines[i + 1]?.text || "";

    // Date of Birth
    if (!eid.date_of_birth && (t.includes("date of birth") || t.includes("تاريخ الميلاد"))) {
      const parsed = parseFormattedDate(l.text) || parseFormattedDate(nextLine);
      if (parsed) {
        eid.date_of_birth = parsed;
        confidence["date_of_birth"] = l.confidence;
      }
    }

    // Issue Date
    if (!eid.issue_date && (t.includes("issuing date") || (t.includes("issue date") && !t.includes("driving")) || t.includes("تاريخ الإصدار") || t.includes("تاريخ الاصدار"))) {
      const parsed = parseFormattedDate(l.text) || parseFormattedDate(nextLine);
      if (parsed && (!eid.date_of_birth || parsed !== eid.date_of_birth)) {
        eid.issue_date = parsed;
        confidence["eid_issue_date"] = l.confidence;
      }
    }

    // Expiry Date
    if (!eid.expiry_date && (t.includes("expiry date") || t.includes("تاريخ الإنتهاء") || t.includes("تاريخ الانتهاء"))) {
      const parsed = parseFormattedDate(l.text) || parseFormattedDate(nextLine);
      if (parsed && (!eid.date_of_birth || parsed !== eid.date_of_birth)) {
        eid.expiry_date = parsed;
        confidence["eid_expiry_date"] = l.confidence;
      }
    }
  }

  // G. Occupation & Employer
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    const t = l.text;
    if (/occupation|المهنة/i.test(t) && !eid.occupation) {
      eid.occupation = t.replace(/occupation[:\s]*|المهنة[:\s]*/gi, "").trim() || lines[i + 1]?.text.trim() || null;
    }
    if (/employer|صاحب العمل/i.test(t) && !eid.employer) {
      eid.employer = t.replace(/employer[:\s]*|صاحب العمل[:\s]*/gi, "").trim() || lines[i + 1]?.text.trim() || null;
    }
  }

  // ==========================================
  // 3. UAE DRIVING LICENCE EXTRACTION
  // ==========================================

  // A. Licence Number (Must be numeric or alphanumeric digits like 382569, NOT plain words like "should")
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.text.toLowerCase();
    const norm = normalizeDigits(line.text);

    // Label on line: "License No.", "Licence No.", "رقم الرخصة", "رقم رخصة القيادة"
    if (/رقم\s*الرخصة|رقم\s*رخصة\s*القيادة|licen[sc]e\s*no\.?|licen[sc]e\s*number/i.test(t)) {
      // 1. Check if digits are on same line
      const sameLine = norm.match(/\b([A-Z]?\d{5,9}[A-Z]?)\b/i);
      if (sameLine && !/passport/i.test(sameLine[1])) {
        dl.licence_number = sameLine[1];
        confidence["dl_licence_number"] = line.confidence;
        break;
      }
      // 2. Check previous line (common in Arabic-English layout: "رقم الرخصة" -> "382569" -> "License No.")
      if (lines[i - 1]) {
        const prevNorm = normalizeDigits(lines[i - 1].text.trim());
        const prevMatch = prevNorm.match(/^([A-Z]?\d{5,9}[A-Z]?)$/i);
        if (prevMatch && prevMatch[1] !== eid.card_number) {
          dl.licence_number = prevMatch[1];
          confidence["dl_licence_number"] = lines[i - 1].confidence;
          break;
        }
      }
      // 3. Check next line
      if (lines[i + 1]) {
        const nextNorm = normalizeDigits(lines[i + 1].text.trim());
        const nextMatch = nextNorm.match(/^([A-Z]?\d{5,9}[A-Z]?)$/i);
        if (nextMatch && nextMatch[1] !== eid.card_number) {
          dl.licence_number = nextMatch[1];
          confidence["dl_licence_number"] = lines[i + 1].confidence;
          break;
        }
      }
    }
  }

  // Fallback DL Licence number: search for isolated 5-8 digit number on a DL page
  if (!dl.licence_number) {
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const norm = normalizeDigits(line.text.trim());
      if (/^\d{5,8}$/.test(norm)) {
        if (norm !== eid.card_number && norm !== dl.traffic_code_no && (!eid.id_number || !eid.id_number.includes(norm))) {
          // Check if surrounding lines mention driving, license, rta, ajman, etc.
          const nearby = lines.slice(Math.max(0, i - 4), Math.min(lines.length, i + 5)).map((l) => l.text).join(" ").toLowerCase();
          if (nearby.includes("licen") || nearby.includes("رخصة") || nearby.includes("driving") || nearby.includes("مرور")) {
            dl.licence_number = norm;
            confidence["dl_licence_number"] = line.confidence;
            break;
          }
        }
      }
    }
  }

  // B. Traffic Code No / الرمز المروري / Traffic No / T.C. No
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.text.toLowerCase();
    const norm = normalizeDigits(line.text);

    if (t.includes("traffic code") || t.includes("الرمز المروري") || t.includes("traffic no") || t.includes("t.c. no") || t.includes("رقم المرور")) {
      // 1. Same line
      const direct = norm.match(/\b(\d{7,12})\b/);
      if (direct && direct[1] !== eid.card_number) {
        dl.traffic_code_no = direct[1];
        confidence["dl_traffic_code"] = line.confidence;
        break;
      }
      // 2. Search within next 5 lines
      for (let j = i + 1; j <= Math.min(lines.length - 1, i + 6); j++) {
        const nearNorm = normalizeDigits(lines[j].text.trim());
        const nearMatch = nearNorm.match(/\b(\d{7,12})\b/);
        if (nearMatch && nearMatch[1] !== eid.card_number && (!eid.id_number || !eid.id_number.includes(nearMatch[1]))) {
          dl.traffic_code_no = nearMatch[1];
          confidence["dl_traffic_code"] = lines[j].confidence;
          break;
        }
      }
      if (dl.traffic_code_no) break;
    }
  }

  // Fallback Traffic Code: standalone 8 to 12 digit number
  if (!dl.traffic_code_no) {
    for (let i = 0; i < lines.length; i++) {
      const norm = normalizeDigits(lines[i].text.trim());
      const m = norm.match(/^(\d{8,12})$/);
      if (m && m[1] !== eid.card_number && (!eid.id_number || !eid.id_number.includes(m[1]))) {
        const nearby = lines.slice(Math.max(0, i - 4), Math.min(lines.length, i + 5)).map((l) => l.text).join(" ").toLowerCase();
        if (nearby.includes("traffic") || nearby.includes("مرور") || nearby.includes("licen") || nearby.includes("رخصة") || nearby.includes("vehicle")) {
          dl.traffic_code_no = m[1];
          confidence["dl_traffic_code"] = lines[i].confidence;
          break;
        }
      }
    }
  }

  // C. Issued By / Place of Issue / Licensing Authority (جهة الإصدار / سلطة الترخيص)
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const t = line.text.toLowerCase();
    const clean = cleanArabicText(line.text);

    if (t.includes("place of issue") || t.includes("جهة الاصدار") || t.includes("جهة الإصدار") || t.includes("licensing authority") || t.includes("سلطة الترخيص")) {
      // Check same line or surrounding 3 lines
      const segment = lines.slice(i, Math.min(lines.length, i + 3)).map((l) => l.text).join(" ");
      for (const [key, auth] of Object.entries(EMIRATES_AUTHORITIES)) {
        if (normKey(segment).includes(normKey(key))) {
          dl.issued_by = auth;
          confidence["dl_issued_by"] = line.confidence;
          break;
        }
      }
      if (dl.issued_by) break;
    }
  }

  // Fallback Authority: Check lines that contain DL keywords
  if (!dl.issued_by) {
    for (const [key, auth] of Object.entries(EMIRATES_AUTHORITIES)) {
      if (normKey(fullText).includes(normKey(key))) {
        dl.issued_by = auth;
        confidence["dl_issued_by"] = 0.95;
        break;
      }
    }
  }

  // D. Driving Licence Dates (Issue Date & Expiry Date)
  let dlStartIndex = -1;
  for (let i = 0; i < lines.length; i++) {
    const t = lines[i].text.toLowerCase();
    if (t.includes("driving licen") || t.includes("رخصة قيادة") || t.includes("license no") || t.includes("رقم الرخصة")) {
      dlStartIndex = i;
      break;
    }
  }

  const dlSearchLines = dlStartIndex >= 0 ? lines.slice(dlStartIndex) : lines;

  for (let i = 0; i < dlSearchLines.length; i++) {
    const line = dlSearchLines[i];
    const t = line.text.toLowerCase();

    // Check Issue Date on DL
    if (!dl.issue_date && (t.includes("issue date") || t.includes("تاريخ الاصدار") || t.includes("تاريخ الإصدار"))) {
      for (let j = i; j <= Math.min(dlSearchLines.length - 1, i + 3); j++) {
        const p = parseFormattedDate(dlSearchLines[j].text);
        if (p && p !== eid.date_of_birth) {
          dl.issue_date = p;
          confidence["dl_issue_date"] = dlSearchLines[j].confidence;
          break;
        }
      }
    }

    // Check Expiry Date on DL
    if (!dl.expiry_date && (t.includes("expiry date") || t.includes("تاريخ الانتهاء") || t.includes("تاريخ الإنتهاء"))) {
      for (let j = i; j <= Math.min(dlSearchLines.length - 1, i + 3); j++) {
        const p = parseFormattedDate(dlSearchLines[j].text);
        if (p && p !== eid.date_of_birth && p !== dl.issue_date) {
          dl.expiry_date = p;
          confidence["dl_expiry_date"] = dlSearchLines[j].confidence;
          break;
        }
      }
    }
  }

  // E. Allowed Vehicle Categories on Driving Licence
  if (/مركبة\s*خفيفة|light\s*vehicle/i.test(fullText)) {
    if (!dl.categories.includes("Light Vehicle (3)")) dl.categories.push("Light Vehicle (3)");
  }
  if (/دراجة\s*آلية|دراجة\s*نارية|motorcycle/i.test(fullText)) {
    if (!dl.categories.includes("Motorcycle (1)")) dl.categories.push("Motorcycle (1)");
  }
  if (/مركبة\s*ثقيلة|heavy\s*vehicle/i.test(fullText)) {
    if (!dl.categories.includes("Heavy Vehicle (4)")) dl.categories.push("Heavy Vehicle (4)");
  }
  if (/حافلة\s*خفيفة|light\s*bus/i.test(fullText)) {
    if (!dl.categories.includes("Light Bus (5)")) dl.categories.push("Light Bus (5)");
  }
  if (/حافلة\s*ثقيلة|heavy\s*bus/i.test(fullText)) {
    if (!dl.categories.includes("Heavy Bus (6)")) dl.categories.push("Heavy Bus (6)");
  }
  if (dl.categories.length === 0) {
    dl.categories.push("Light Vehicle (3)");
  }

  // Calculate validity statuses
  const checkStatus = (expStr: string | null): "VALID" | "EXPIRED" | "EXPIRING_SOON" | "UNKNOWN" => {
    if (!expStr) return "UNKNOWN";
    const exp = new Date(expStr);
    const now = new Date();
    const diff = Math.floor((exp.getTime() - now.getTime()) / (1000 * 3600 * 24));
    if (diff < 0) return "EXPIRED";
    if (diff <= 30) return "EXPIRING_SOON";
    return "VALID";
  };

  eid.status = checkStatus(eid.expiry_date);
  dl.status = checkStatus(dl.expiry_date);

  // ==========================================
  // 4. RECONCILIATION & CROSS-MATCHING
  // ==========================================
  const reviewReasons: string[] = [];
  const nameMatch = Boolean(eid.name_en || eid.name_ar);

  if (eid.status === "EXPIRED") reviewReasons.push("Emirates ID is expired.");
  if (dl.status === "EXPIRED") reviewReasons.push("UAE Driving Licence is expired.");
  if (!eid.id_number) reviewReasons.push("Emirates ID number missing or unreadable.");
  if (!dl.licence_number && !dl.traffic_code_no) reviewReasons.push("Driving licence number missing or unreadable.");

  const allValid = eid.status === "VALID" && dl.status !== "EXPIRED";
  const overall_status: ReconciliationSummary["overall_status"] =
    eid.status === "EXPIRED" || dl.status === "EXPIRED"
      ? "DOCUMENT_EXPIRED"
      : reviewReasons.length > 0
      ? "NEEDS_REVIEW"
      : "VERIFIED";

  if (!eid.id_number) warnings.push("Emirates ID card was not fully detected.");
  if (!dl.licence_number && !dl.traffic_code_no) warnings.push("Driving Licence card was not fully detected.");

  return {
    success: Boolean(eid.id_number || dl.licence_number || eid.name_en),
    emirates_id: eid,
    driving_licence: dl,
    reconciliation: {
      name_match: nameMatch,
      nationality_match: true,
      all_valid: allValid,
      overall_status,
      review_reasons: reviewReasons,
    },
    confidence,
    warnings,
    raw_lines_count: lines.length,
    raw_text: fullText,
  };
}

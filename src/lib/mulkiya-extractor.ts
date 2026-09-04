import { DocumentAIResponse, ExtractedLine } from "./documentai";

export interface MulkiyaData {
  plate_source: string | null;
  plate_category: string | null;
  plate_code: string | null;
  plate_number: string | null;
  vin: string | null;
  make: string | null;
  model: string | null;
  year: number | null;
  color: string | null;
  insurance_company: string | null;
  policy_number: string | null;
  insurance_expiry: string | null;
  registration_expiry: string | null;
  registration_issuance: string | null;
}

export interface ExtractionResult {
  success: boolean;
  data: MulkiyaData;
  confidence: Record<string, number | null>;
  warnings: string[];
  raw_lines_count: number;
  raw_text: string;
}

// Arabic-Indic to ASCII digits translation
const ARABIC_INDIC_DIGITS: Record<string, string> = {
  "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
  "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
  "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
  "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
};

export function normalizeDigits(text: string): string {
  return text.replace(/[٠-٩۰-۹]/g, (ch) => ARABIC_INDIC_DIGITS[ch] || ch);
}

export function foldArabic(text: string): string {
  return text
    .replace(/[ً-ْٰـ]/g, "") // remove tashkeel & tatweel
    .replace(/[أإآٱ]/g, "ا")
    .replace(/ى/g, "ي")
    .replace(/ة/g, "ه")
    .replace(/ؤ/g, "و")
    .replace(/ئ/g, "ي");
}

export function normKey(text: string): string {
  const folded = foldArabic(normalizeDigits(text).toLowerCase());
  return folded.replace(/[^a-z0-9\u0600-\u06ff]+/g, " ").trim();
}

// Emirates Dictionary
const EMIRATES: Record<string, string> = {
  "dubai": "Dubai", "دبي": "Dubai",
  "abu dhabi": "Abu Dhabi", "ابو ظبي": "Abu Dhabi", "ابوظبي": "Abu Dhabi",
  "sharjah": "Sharjah", "الشارقه": "Sharjah", "الشارقة": "Sharjah",
  "ajman": "Ajman", "عجمان": "Ajman",
  "umm al quwain": "Umm Al Quwain", "أم القيوين": "Umm Al Quwain", "ام القيوين": "Umm Al Quwain",
  "ras al khaimah": "Ras Al Khaimah", "راس الخيمه": "Ras Al Khaimah", "رأس الخيمة": "Ras Al Khaimah",
  "fujairah": "Fujairah", "الفجيره": "Fujairah", "الفجيرة": "Fujairah",
};

// Categories Dictionary
const CATEGORIES: Record<string, string> = {
  "private": "Private", "خصوصي": "Private",
  "commercial": "Commercial", "تجاري": "Commercial",
  "taxi": "Taxi", "أجرة": "Taxi", "اجره": "Taxi",
  "export": "Export", "تصدير": "Export",
  "classic": "Classic", "كلاسيك": "Classic",
};

// Colors Dictionary
const COLORS: Record<string, string> = {
  "white": "White", "ابيض": "White", "أبيض": "White",
  "black": "Black", "اسود": "Black", "أسود": "Black",
  "grey": "Grey", "gray": "Grey", "رمادي": "Grey",
  "silver": "Silver", "فضي": "Silver",
  "red": "Red", "احمر": "Red", "أحمر": "Red",
  "blue": "Blue", "ازرق": "Blue", "أزرق": "Blue",
  "green": "Green", "اخضر": "Green", "أخضر": "Green",
  "yellow": "Yellow", "اصفر": "Yellow", "أصفر": "Yellow",
  "brown": "Brown", "بني": "Brown",
  "gold": "Gold", "ذهبي": "Gold",
  "orange": "Orange", "برتقالي": "Orange",
  "beige": "Beige", "بيج": "Beige",
};

// Popular Makes in UAE
const LUXURY_MAKES = [
  "ROLLS-ROYCE", "BENTLEY", "FERRARI", "LAMBORGHINI", "PORSCHE",
  "MERCEDES-BENZ", "MERCEDES", "BMW", "AUDI", "ASTON MARTIN", "MCLAREN",
  "RANGE ROVER", "LAND ROVER", "MASERATI", "CADILLAC", "LEXUS",
  "JAGUAR", "CHEVROLET", "FORD", "NISSAN", "TOYOTA", "HYUNDAI", "KIA",
  "VOLKSWAGEN", "TESLA", "JEEP", "DODGE", "GMC"
];

const DATE_REGEX = /(?<!\d)([0-3]?\d)[\/\-.]([01]?\d)[\/\-.]((?:19|20)?\d{2})(?!\d)/g;
const YEAR_REGEX = /\b(19[89]\d|20[0-3]\d)\b/g;
const VIN_REGEX = /\b[A-HJ-NPR-Z0-9]{17}\b/gi;

function parseDateMatch(d: string, m: string, y: string): string {
  const day = d.padStart(2, "0");
  const month = m.padStart(2, "0");
  let year = y;
  if (year.length === 2) {
    const num = parseInt(year, 10);
    year = num > 50 ? `19${year}` : `20${year}`;
  }
  return `${year}-${month}-${day}`;
}

export function extractMulkiyaFields(doc: DocumentAIResponse): ExtractionResult {
  const warnings: string[] = [];
  const confidence: Record<string, number | null> = {};

  const fullText = normalizeDigits(doc.rawText);
  const lines = doc.lines;

  const data: MulkiyaData = {
    plate_source: null,
    plate_category: null,
    plate_code: null,
    plate_number: null,
    vin: null,
    make: null,
    model: null,
    year: null,
    color: null,
    insurance_company: null,
    policy_number: null,
    insurance_expiry: null,
    registration_expiry: null,
    registration_issuance: null,
  };

  // 1. VIN (Chassis No) Extraction
  // Priority A: Look near chassis labels
  let vinCandidate: string | null = null;
  const chassisLabelIndices: number[] = [];
  lines.forEach((l, idx) => {
    const t = l.text.toLowerCase();
    if (t.includes("chassis") || t.includes("vin") || t.includes("القاعدة") || t.includes("الشاسي")) {
      chassisLabelIndices.push(idx);
    }
  });

  // Check lines adjacent to chassis labels (same line or next 2 lines)
  for (const idx of chassisLabelIndices) {
    const candidateLines = [lines[idx], lines[idx + 1], lines[idx + 2]].filter(Boolean);
    for (const cl of candidateLines) {
      const words = normalizeDigits(cl.text).toUpperCase().split(/\s+/);
      for (const w of words) {
        const clean = w.replace(/[^A-Z0-9]/g, "");
        if (/^[A-HJ-NPR-Z0-9]{17}$/.test(clean) && !clean.includes("EMIRATE") && /\d/.test(clean)) {
          vinCandidate = clean;
          confidence["vin"] = cl.confidence;
          break;
        }
      }
      if (vinCandidate) break;
    }
    if (vinCandidate) break;
  }

  // Priority B: Scan all lines if not found near label
  if (!vinCandidate) {
    for (const line of lines) {
      const words = normalizeDigits(line.text).toUpperCase().split(/\s+/);
      for (const w of words) {
        const clean = w.replace(/[^A-Z0-9]/g, "");
        if (/^[A-HJ-NPR-Z0-9]{17}$/.test(clean) && !clean.includes("EMIRATE") && /\d/.test(clean) && /[A-Z]/.test(clean)) {
          vinCandidate = clean;
          confidence["vin"] = line.confidence;
          break;
        }
      }
      if (vinCandidate) break;
    }
  }
  data.vin = vinCandidate;

  // 2. Plate Source (Emirate)
  for (const line of lines) {
    const key = normKey(line.text);
    for (const [k, v] of Object.entries(EMIRATES)) {
      if (key.includes(normKey(k))) {
        data.plate_source = v;
        confidence["plate_source"] = line.confidence;
        break;
      }
    }
    if (data.plate_source) break;
  }

  // 3. Plate Category
  for (const line of lines) {
    const key = normKey(line.text);
    for (const [k, v] of Object.entries(CATEGORIES)) {
      if (key.includes(normKey(k))) {
        data.plate_category = v;
        confidence["plate_category"] = line.confidence;
        break;
      }
    }
    if (data.plate_category) break;
  }

  // 4. Plate Number & Code
  // In Dubai Mulkiya: often formatted like "AA / 88271" or "A 12345" or "12 / 34567"
  for (const line of lines) {
    const norm = normalizeDigits(line.text);
    // Pattern: 1-3 letters / digits followed by plate number
    const plateMatch = norm.match(/\b([A-Z]{1,3})\s*[\/|\-]?\s*(\d{1,6})\b/i);
    if (plateMatch && !plateMatch[0].includes(data.vin || "XYZ") && !plateMatch[0].includes("TC")) {
      data.plate_code = plateMatch[1].toUpperCase();
      data.plate_number = plateMatch[2];
      confidence["plate_number"] = line.confidence;
      confidence["plate_code"] = line.confidence;
      break;
    }
    // Just digits next to "Plate No" or "رقم اللوحة"
    const digitMatch = norm.match(/\b(\d{4,6})\b/);
    if (digitMatch && !data.plate_number && (!data.vin || !data.vin.includes(digitMatch[1]))) {
      data.plate_number = digitMatch[1];
    }
  }

  // 5. Make & Model
  for (const make of LUXURY_MAKES) {
    const re = new RegExp(`\\b${make.replace("-", "[-\\s]?")}\\b`, "i");
    const foundLine = lines.find((l) => re.test(l.text));
    if (foundLine) {
      data.make = make;
      confidence["make"] = foundLine.confidence;

      // Extract model candidate from same line
      const afterMake = foundLine.text.replace(re, "").trim();
      if (afterMake.length >= 2 && !/^\d+$/.test(afterMake)) {
        data.model = afterMake.split(/[,/]/)[0].trim();
        confidence["model"] = foundLine.confidence;
      }
      break;
    }
  }

  // 6. Model Year (Look next to "Model" or "سنة الصنع" first)
  for (const line of lines) {
    const norm = normalizeDigits(line.text);
    const mMatch = norm.match(/(?:model|سنة الصنع|الموديل)\s*[:\-]?\s*(20[0-3]\d|19[89]\d)/i);
    if (mMatch) {
      data.year = parseInt(mMatch[1], 10);
      confidence["year"] = line.confidence;
      break;
    }
  }

  if (!data.year) {
    const yearMatches: number[] = [];
    let yMatch: RegExpExecArray | null;
    const yRegex = new RegExp(YEAR_REGEX);
    while ((yMatch = yRegex.exec(fullText)) !== null) {
      const yVal = parseInt(yMatch[1], 10);
      if (yVal >= 1990 && yVal <= new Date().getFullYear() + 1) {
        yearMatches.push(yVal);
      }
    }
    if (yearMatches.length > 0) {
      data.year = Math.max(...yearMatches);
      confidence["year"] = 0.88;
    }
  }

  // 7. Color
  for (const line of lines) {
    const key = normKey(line.text);
    for (const [k, v] of Object.entries(COLORS)) {
      if (key.includes(normKey(k))) {
        data.color = v;
        confidence["color"] = line.confidence;
        break;
      }
    }
    if (data.color) break;
  }

  // 8. Dates (Expiry, Issuance, Insurance Expiry)
  const allDates: string[] = [];
  let match: RegExpExecArray | null;
  const dRegex = new RegExp(DATE_REGEX);
  while ((match = dRegex.exec(fullText)) !== null) {
    const parsed = parseDateMatch(match[1], match[2], match[3]);
    if (!allDates.includes(parsed)) {
      allDates.push(parsed);
    }
  }

  // Sort dates chronologically
  allDates.sort();

  if (allDates.length >= 3) {
    data.registration_issuance = allDates[0];
    data.registration_expiry = allDates[1];
    data.insurance_expiry = allDates[2];
  } else if (allDates.length === 2) {
    data.registration_issuance = allDates[0];
    data.registration_expiry = allDates[1];
    data.insurance_expiry = allDates[1];
  } else if (allDates.length === 1) {
    data.registration_expiry = allDates[0];
  }

  // 9. Insurance Company & Policy Number
  lines.forEach((line, idx) => {
    const text = line.text;
    const lower = text.toLowerCase();
    
    // Insurance Company
    if (
      lower.includes("insurance") ||
      text.includes("تأمين") ||
      text.includes("انشورنس") ||
      text.includes("مؤمنة")
    ) {
      let cleaned = text
        .replace(/(insurance|company|co\.|تأمين|انشورنس|مؤمنة لدى|نوع التأمين|شامل|إنتهاء التأمين|انتهاء التأمين)/gi, "")
        .replace(/\(.*$/, "")
        .replace(/فرع.*$/, "")
        .trim();
      if (cleaned.length > 3 && !data.insurance_company) {
        data.insurance_company = cleaned;
        confidence["insurance_company"] = line.confidence;
      }
    }

    // Policy Number
    if (lower.includes("policy") || text.includes("وثيقة") || text.includes("بوليصة")) {
      // Check current line and next line
      const currentAndNext = [line.text, lines[idx + 1]?.text].join(" ");
      const numMatch = currentAndNext.match(/\b(\d{8,14})\b/);
      if (numMatch && !data.policy_number) {
        data.policy_number = numMatch[1];
        confidence["policy_number"] = line.confidence;
      }
    }
  });

  // Generate warnings for missing critical fields
  if (!data.vin) warnings.push("Chassis / VIN number could not be detected clearly.");
  if (!data.plate_number) warnings.push("Plate number was not identified.");
  if (!data.registration_expiry) warnings.push("Registration expiry date was not found.");

  return {
    success: Boolean(data.vin || data.plate_number || data.make),
    data,
    confidence,
    warnings,
    raw_lines_count: lines.length,
    raw_text: fullText,
  };
}

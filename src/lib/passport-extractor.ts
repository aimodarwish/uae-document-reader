import { DocumentAIResponse } from "./documentai";
import { normalizeDigits } from "./mulkiya-extractor";

export interface PassportData {
  passport_number: string | null;
  first_name: string | null;
  last_name: string | null;
  full_name: string | null;
  nationality_code: string | null;
  nationality_name: string | null;
  issuing_country: string | null;
  date_of_birth: string | null;
  gender: "M" | "F" | "X" | null;
  issue_date: string | null;
  expiry_date: string | null;
  mrz_lines: string[];
  mrz_valid: boolean;
  status: "VALID" | "EXPIRED" | "EXPIRING_SOON" | "UNKNOWN";
}

export interface PassportExtractionResult {
  success: boolean;
  data: PassportData;
  confidence: Record<string, number | null>;
  warnings: string[];
  raw_lines_count: number;
  raw_text: string;
}

const ISO3_COUNTRIES: Record<string, string> = {
  ARE: "United Arab Emirates",
  IRL: "Ireland",
  GBR: "United Kingdom",
  USA: "United States",
  SAU: "Saudi Arabia",
  KWT: "Kuwait",
  BHR: "Bahrain",
  QAT: "Qatar",
  OMN: "Oman",
  IND: "India",
  PAK: "Pakistan",
  PHL: "Philippines",
  EGY: "Egypt",
  JOR: "Jordan",
  LBN: "Lebanon",
  SYR: "Syria",
  RUS: "Russia",
  DEU: "Germany",
  FRA: "France",
  ITA: "Italy",
  ESP: "Spain",
  CAN: "Canada",
  AUS: "Australia",
  CHN: "China",
  TUR: "Turkey",
  MAR: "Morocco",
  TUN: "Tunisia",
  DZA: "Algeria",
  IRN: "Iran",
  SGP: "Singapore",
  CHE: "Switzerland",
  NLD: "Netherlands",
  BEL: "Belgium",
  SWE: "Sweden",
  NOR: "Norway",
  DNK: "Denmark",
  FIN: "Finland",
  AUT: "Austria",
  POL: "Poland",
  UKR: "Ukraine",
  ZAF: "South Africa",
  NZL: "New Zealand",
  BRA: "Brazil",
  ARG: "Argentina",
  MEX: "Mexico",
  JPN: "Japan",
  KOR: "South Korea",
};

// Check if a line is genuinely an MRZ line (must have '<' or TD3 structure)
function cleanMrzLine(line: string): string {
  // Preserve original '<' and letters/digits, convert spaces to '<' only if line already has '<'
  if (!line.includes("<") && !line.startsWith("P")) return "";
  return line.toUpperCase().replace(/\s+/g, "<").replace(/[^A-Z0-9<]/g, "");
}

function parseYYMMDD(raw: string, isBirth: boolean): string | null {
  if (raw.length !== 6 || !/^\d{6}$/.test(raw)) return null;
  const yy = parseInt(raw.substring(0, 2), 10);
  const mm = raw.substring(2, 4);
  const dd = raw.substring(4, 6);
  const currentYY = new Date().getFullYear() % 100;

  let century = 2000;
  if (isBirth) {
    century = yy > currentYY ? 1900 : 2000;
  } else {
    // For expiry / issue dates
    century = yy > currentYY + 20 ? 1900 : 2000;
  }
  const year = century + yy;
  return `${year}-${mm}-${dd}`;
}

export function extractPassportFields(doc: DocumentAIResponse): PassportExtractionResult {
  const warnings: string[] = [];
  const confidence: Record<string, number | null> = {};
  const fullText = normalizeDigits(doc.rawText);

  const data: PassportData = {
    passport_number: null,
    first_name: null,
    last_name: null,
    full_name: null,
    nationality_code: null,
    nationality_name: null,
    issuing_country: null,
    date_of_birth: null,
    gender: null,
    issue_date: null,
    expiry_date: null,
    mrz_lines: [],
    mrz_valid: false,
    status: "UNKNOWN",
  };

  // 1. Scan for TD3 (44 chars) or TD2 (36 chars) MRZ lines
  // A genuine passport MRZ line 1 starts with P< and has consecutive chevrons
  let line1 = "";
  let line2 = "";

  // Priority 1: Search raw lines from bottom up (MRZ is always at the bottom of the card)
  for (let i = doc.lines.length - 1; i >= 0; i--) {
    const rawText = doc.lines[i].text.toUpperCase().replace(/\s+/g, "");
    // Check if it matches Line 1 of passport: P<XXX... with at least 3 '<'
    if (/^P<[A-Z]{3}/.test(rawText) && rawText.includes("<<")) {
      line1 = rawText;
      // Line 2 is typically the line right after it, or lines[i + 1]
      if (doc.lines[i + 1]) {
        line2 = doc.lines[i + 1].text.toUpperCase().replace(/\s+/g, "");
      }
      break;
    }
  }

  // Priority 2: If not found, scan lines that have P< and multiple '<'
  if (!line1) {
    for (let i = 0; i < doc.lines.length; i++) {
      const cleaned = cleanMrzLine(doc.lines[i].text);
      if (/^P[<A-Z][A-Z]{3}/.test(cleaned) && cleaned.includes("<<") && cleaned.length >= 25) {
        line1 = cleaned;
        if (doc.lines[i + 1]) {
          line2 = cleanMrzLine(doc.lines[i + 1].text);
        }
        break;
      }
    }
  }

  if (line1) data.mrz_lines.push(line1);
  if (line2) data.mrz_lines.push(line2);

  // Parse Line 1: P<ISSUER SURNAME<<GIVEN<NAMES<<<<
  if (line1) {
    const issuerMatch = line1.match(/^P[A-Z<]([A-Z]{3})/);
    if (issuerMatch) {
      data.issuing_country = ISO3_COUNTRIES[issuerMatch[1]] || issuerMatch[1];
    }

    const nameSection = line1.substring(5);
    const parts = nameSection.split("<<");
    if (parts.length >= 2) {
      const surname = parts[0].replace(/</g, " ").trim();
      const given = parts[1].replace(/</g, " ").trim();
      data.last_name = surname || null;
      data.first_name = given || null;
      data.full_name = [given, surname].filter(Boolean).join(" ") || null;
      confidence["full_name"] = 0.96;
    }
  }

  // Parse Line 2: DOC_NUM+CHECK NATIONALITY DOB+CHECK GENDER EXPIRY+CHECK
  if (line2 && line2.length >= 28) {
    // Document number (first 9 characters)
    const rawDocNum = line2.substring(0, 9).replace(/</g, "").trim();
    if (rawDocNum) {
      data.passport_number = rawDocNum;
      confidence["passport_number"] = 0.98;
    }

    // Nationality (chars 10..13)
    const natCode = line2.substring(10, 13).replace(/</g, "").trim();
    if (natCode.length === 3) {
      data.nationality_code = natCode;
      data.nationality_name = ISO3_COUNTRIES[natCode] || natCode;
      confidence["nationality"] = 0.97;
    }

    // Date of birth (chars 13..19)
    const rawDob = line2.substring(13, 19);
    data.date_of_birth = parseYYMMDD(rawDob, true);

    // Gender (char 20)
    const gChar = line2.charAt(20).toUpperCase();
    if (gChar === "M" || gChar === "F") {
      data.gender = gChar;
    }

    // Expiry date (chars 21..27)
    const rawExp = line2.substring(21, 27);
    data.expiry_date = parseYYMMDD(rawExp, false);
    if (data.expiry_date) confidence["expiry_date"] = 0.97;

    data.mrz_valid = Boolean(data.passport_number && data.full_name);
  }

  // Visual Inspection Fallback if MRZ missed some fields
  if (!data.passport_number) {
    // A. Look directly next to passport labels
    for (const l of doc.lines) {
      const t = l.text;
      if (/(?:passport|pass|doc|document)\s*(?:no|num|number)?|رقم الجواز/i.test(t)) {
        const m = t.match(/\b([A-Z0-9]{7,10})\b/i);
        if (m && !/passport|number/i.test(m[1])) {
          data.passport_number = m[1].toUpperCase();
          confidence["passport_number"] = l.confidence;
          break;
        }
      }
    }
    // B. Look for standalone passport number pattern
    if (!data.passport_number) {
      for (const l of doc.lines) {
        const m = l.text.trim().match(/^([A-Z]\d{7,8}|\d{8,9})$/);
        if (m && !data.date_of_birth?.includes(m[1])) {
          data.passport_number = m[1];
          confidence["passport_number"] = l.confidence;
          break;
        }
      }
    }
  }

  if (!data.full_name) {
    // Scan for name labels
    for (let i = 0; i < doc.lines.length; i++) {
      const l = doc.lines[i];
      if (/given names|given name|surname|nom|nombre|الاسم/i.test(l.text) && !/father|mother/i.test(l.text)) {
        const direct = l.text.replace(/given names?|surname|nom|nombre|الاسم[:\s]*/gi, "").trim();
        if (direct.length > 2 && /^[A-Za-z\s]+$/.test(direct)) {
          data.full_name = (data.full_name ? `${data.full_name} ${direct}` : direct).trim();
        } else if (doc.lines[i + 1]) {
          const next = doc.lines[i + 1].text.trim();
          if (next.length > 3 && /^[A-Za-z\s]+$/.test(next) && !/passport|republic|kingdom/i.test(next)) {
            data.full_name = (data.full_name ? `${data.full_name} ${next}` : next).trim();
          }
        }
      }
    }

    // Fallback: search for prominent capitalized multi-word line
    if (!data.full_name) {
      for (const l of doc.lines) {
        const t = l.text.trim();
        const words = t.split(/\s+/);
        if (
          words.length >= 2 &&
          words.length <= 4 &&
          /^[A-Z\s]+$/.test(t) &&
          !/PASSPORT|REPUBLIC|KINGDOM|UNITED|STATES|MINISTRY|AUTHORITY|FEDERAL/i.test(t)
        ) {
          data.full_name = t;
          confidence["full_name"] = 0.85;
          break;
        }
      }
    }
  }

  // Nationality fallback from common adjectives
  if (!data.nationality_name) {
    const natKeywords: Record<string, string> = {
      EMIRATI: "United Arab Emirates",
      BRITISH: "United Kingdom",
      AMERICAN: "United States",
      FRENCH: "France",
      GERMAN: "Germany",
      ITALIAN: "Italy",
      RUSSIAN: "Russia",
      INDIAN: "India",
      PAKISTANI: "Pakistan",
      SAUDI: "Saudi Arabia",
      EGYPTIAN: "Egypt",
      CANADIAN: "Canada",
      AUSTRALIAN: "Australia",
      SPANISH: "Spain",
      CHINESE: "China",
      FILIPINO: "Philippines",
      JORDANIAN: "Jordan",
      LEBANESE: "Lebanon",
      SYRIAN: "Syria",
    };
    for (const [adjective, country] of Object.entries(natKeywords)) {
      if (new RegExp(`\\b${adjective}\\b`, "i").test(fullText)) {
        data.nationality_name = country;
        break;
      }
    }
  }

  // Find dates from text if not in MRZ (both numeric DD/MM/YYYY and word dates like 25 APR 2023)
  const MONTHS_MAP: Record<string, string> = {
    JAN: "01", FEB: "02", MAR: "03", APR: "04", MAY: "05", JUN: "06",
    JUL: "07", AUG: "08", SEP: "09", OCT: "10", NOV: "11", DEC: "12"
  };

  const allDates: string[] = [];

  // Numeric dates
  const dateRegex = /(?<!\d)([0-3]?\d)[\/\-.]([01]?\d)[\/\-.]((?:19|20)?\d{2})(?!\d)/g;
  let dMatch: RegExpExecArray | null;
  while ((dMatch = dateRegex.exec(fullText)) !== null) {
    const day = dMatch[1].padStart(2, "0");
    const month = dMatch[2].padStart(2, "0");
    let yr = dMatch[3];
    if (yr.length === 2) yr = `20${yr}`;
    const formatted = `${yr}-${month}-${day}`;
    if (!allDates.includes(formatted)) allDates.push(formatted);
  }

  // Word dates (e.g. 10 FEB 2001, 25 AIB/APR 2023, 24 APR 2033)
  const wordDateRegex = /\b([0-3]?\d)\s+(?:[A-Z]{2,4}\/)?(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*\s+((?:19|20)\d{2})\b/gi;
  while ((dMatch = wordDateRegex.exec(fullText)) !== null) {
    const day = dMatch[1].padStart(2, "0");
    const month = MONTHS_MAP[dMatch[2].toUpperCase()] || "01";
    const yr = dMatch[3];
    const formatted = `${yr}-${month}-${day}`;
    if (!allDates.includes(formatted)) allDates.push(formatted);
  }

  allDates.sort();

  if (allDates.length >= 3) {
    if (!data.date_of_birth) data.date_of_birth = allDates[0];
    if (!data.issue_date) data.issue_date = allDates[1];
    if (!data.expiry_date) data.expiry_date = allDates[allDates.length - 1];
  } else if (allDates.length === 2) {
    if (!data.issue_date) data.issue_date = allDates[0];
    if (!data.expiry_date) data.expiry_date = allDates[1];
  } else if (allDates.length === 1) {
    if (!data.expiry_date) data.expiry_date = allDates[0];
  }

  // Status calculation
  if (data.expiry_date) {
    const exp = new Date(data.expiry_date);
    const now = new Date();
    const diffDays = Math.floor((exp.getTime() - now.getTime()) / (1000 * 3600 * 24));
    if (diffDays < 0) {
      data.status = "EXPIRED";
      warnings.push(`Passport expired on ${data.expiry_date}.`);
    } else if (diffDays <= 30) {
      data.status = "EXPIRING_SOON";
      warnings.push(`Passport expires in ${diffDays} days.`);
    } else {
      data.status = "VALID";
    }
  }

  if (!data.passport_number) warnings.push("Passport number could not be found.");
  if (!data.full_name) warnings.push("Holder name was not clearly detected.");

  return {
    success: true,
    data,
    confidence,
    warnings,
    raw_lines_count: doc.lines.length,
    raw_text: fullText,
  };
}

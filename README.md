# UAE Document Intelligence Suite 🇦🇪

An all-in-one neural document extraction and verification platform built specifically for UAE automotive, car rental, and identity workflows.

Designed with **Next.js 14**, **TypeScript**, and **Serverless Architecture** with **100% In-Memory Zero-Retention Data Privacy**.

---

## 🚀 Key Modules & Capabilities

### 1. 🚗 UAE Vehicle License Reader (الملكية - Mulkiya)
Extracts structured vehicle and registration data from all 7 UAE Emirates (Dubai, Abu Dhabi, Sharjah, Ajman, RAK, Fujairah, UAQ):
* **Plate Information:** Source (Emirate), Category (Private/Commercial), Plate Code, Plate Number.
* **Vehicle Specifications:** Make, Model, Model Year, Color, Chassis Number (VIN).
* **Registration & Insurance:** Expiry Dates, Issuance Date, Insurance Company, Policy Number.

### 2. 🛂 International Passport Reader (جواز السفر)
Extracts and validates international passports using Machine-Readable Zone (MRZ) parser and visual OCR:
* **MRZ Standards:** TD3 (Passports - 44x2) and TD2 standards.
* **Extracted Fields:** Passport Number, Full Name, First/Last Names, Nationality (Name + ISO Code), Issuing Country, Date of Birth, Gender, Expiry Date.
* **Validation:** Automatic check-digit verification and validity status calculation (`VALID`, `EXPIRING_SOON`, `EXPIRED`).

### 3. 💳 UAE Resident Suite (Emirates ID + UAE Driving Licence)
Processes single or multi-card uploads (up to 4 images concurrently in parallel):
* **Emirates ID (Front & Back):**
  - MRZ TD1 (3-line) extraction for instant 100% accuracy.
  - Emirates ID Number (`784-YYYY-XXXXXXX-X`), Card Number, English Name, Arabic Name, Nationality, Date of Birth, Issue Date, Expiry Date, Occupation, Employer.
* **UAE Driving Licence (Front & Back):**
  - Licence Number, Traffic Code No. (الرمز المروري), Place of Issue / Licensing Authority (Dubai RTA, Ajman, Abu Dhabi, etc.), Issue Date, Expiry Date, Allowed Vehicle Categories (`Light Vehicle (3)`, `Motorcycle (1)`, etc.).
* **Cross-Document Reconciliation:**
  - Automated identity and name cross-matching between Emirates ID and Driving Licence.
  - Expiry status audit and overall verification (`VERIFIED`, `NEEDS_REVIEW`, `DOCUMENT_EXPIRED`).

---

## ⚡ Performance & Privacy Architecture

* **Client-side Canvas Pre-compression:** High-resolution mobile photos are automatically optimized before upload, ensuring lightning-fast round trips under **3.5 seconds**.
* **Parallel Multi-File Processing:** Up to 4 document sides are analyzed concurrently with `Promise.all`.
* **Zero-Disk Retention:** Processing is performed in volatile serverless memory (RAM) and immediately discarded.
* **White-Label UI:** Clean modern design with Emerald & White theme, built with pure Vanilla CSS tokens and Lucide Icons.

---

## 🛠️ Getting Started

### Prerequisites
* **Node.js:** v18+ or v20+
* **Package Manager:** `npm` or `pnpm`

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/aimodarwish/uae-document-reader.git
cd uae-document-reader

# 2. Install dependencies
npm install
```

### Environment Configuration

Create a `.env.local` file in the root directory (or copy from `.env.example`):

```env
GCP_PROJECT_ID=your_gcp_project_id
GCP_PROJECT_NUMBER=your_gcp_project_number
GCP_LOCATION=eu
GCP_PROCESSOR_ID=your_processor_id
GCP_CLIENT_EMAIL=your_service_account_email
GCP_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

### Run Locally

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## ☁️ Deployment on Vercel

This application is ready for 1-click deployment on **Vercel Serverless**:

1. Push your repository to GitHub.
2. Import the repository in [Vercel Dashboard](https://vercel.com).
3. In **Project Settings ➔ Environment Variables**, add the variables defined in `.env.example`:
   - `GCP_PROJECT_ID`
   - `GCP_PROJECT_NUMBER`
   - `GCP_LOCATION`
   - `GCP_PROCESSOR_ID`
   - `GCP_CLIENT_EMAIL`
   - `GCP_PRIVATE_KEY`
4. Click **Deploy**.

---

## 📂 Project Structure

```
├── src/
│   ├── app/
│   │   ├── api/extract/route.ts   # Parallel serverless OCR extraction endpoint
│   │   ├── globals.css            # Emerald & White luxury theme tokens
│   │   ├── layout.tsx             # Root layout and metadata
│   │   └── page.tsx               # 3-in-1 interactive verification dashboard
│   └── lib/
│       ├── documentai.ts          # Google Cloud Document AI client (REST / In-Memory)
│       ├── mulkiya-extractor.ts   # Vehicle Mulkiya parser & normalizers
│       ├── passport-extractor.ts  # MRZ TD3/TD2 passport extraction engine
│       ├── residence-extractor.ts # Emirates ID (TD1) & UAE DL reconciliation engine
│       └── sample-data.ts         # Instant demo presets
├── .env.example
├── .gitignore
├── next.config.mjs
├── package.json
├── tsconfig.json
└── README.md
```

---

## 📄 License
Private & Proprietary.

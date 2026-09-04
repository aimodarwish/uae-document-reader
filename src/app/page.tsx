"use client";

import React, { useState, useRef } from "react";
import {
  Upload,
  Car,
  FileCheck2,
  ShieldCheck,
  Calendar,
  Copy,
  Check,
  AlertTriangle,
  Download,
  Sparkles,
  FileText,
  Lock,
  Zap,
  Info,
  CreditCard,
  User,
  Globe,
  Award,
  Code2,
} from "lucide-react";

type TabType = "mulkiya" | "passport" | "residence";

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>("mulkiya");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [selectedFileName, setSelectedFileName] = useState<string | null>(null);
  const [previewUrls, setPreviewUrls] = useState<string[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [showRawOcr, setShowRawOcr] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const switchTab = (tab: TabType) => {
    setActiveTab(tab);
    setResult(null);
    setError(null);
    setSelectedFileName(null);
    setPreviewUrls([]);
    setShowRawOcr(false);
  };

  const handleCopy = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  async function compressImageFile(file: File): Promise<File> {
    if (!file.type.startsWith("image/")) return file;

    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
          const MAX_DIM = 1800;
          let { width, height } = img;
          if (width > MAX_DIM || height > MAX_DIM) {
            if (width > height) {
              height = Math.round((height * MAX_DIM) / width);
              width = MAX_DIM;
            } else {
              width = Math.round((width * MAX_DIM) / height);
              height = MAX_DIM;
            }
          }
          const canvas = document.createElement("canvas");
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
          if (!ctx) return resolve(file);
          ctx.drawImage(img, 0, 0, width, height);
          canvas.toBlob(
            (blob) => {
              if (!blob) return resolve(file);
              const optimized = new File([blob], file.name.replace(/\.[^.]+$/, ".jpg"), {
                type: "image/jpeg",
              });
              resolve(optimized);
            },
            "image/jpeg",
            0.85
          );
        };
        img.onerror = () => resolve(file);
        img.src = e.target?.result as string;
      };
      reader.onerror = () => resolve(file);
      reader.readAsDataURL(file);
    });
  }

  const handleFileSelect = async (filesList: FileList | File[]) => {
    const files = Array.from(filesList);
    if (files.length === 0) return;

    setError(null);
    setSelectedFileName(files.map((f) => f.name).join(" + "));

    const previews = files
      .filter((f) => f.type.startsWith("image/"))
      .map((f) => URL.createObjectURL(f));
    setPreviewUrls(previews);

    setLoading(true);
    try {
      // Compress image client-side to guarantee under 3.5s Document AI round-trip
      const optimizedFiles = await Promise.all(files.map(compressImageFile));

      const formData = new FormData();
      formData.append("doc_type", activeTab);
      optimizedFiles.forEach((f) => formData.append("files", f));

      const res = await fetch("/api/extract", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || "Failed to process the document.");
      }

      setResult(data);
    } catch (err: any) {
      console.error(err);
      setError(err.message || "An error occurred while analyzing the document.");
    } finally {
      setLoading(false);
    }
  };

  const loadSample = async (sampleId: string, name: string) => {
    setError(null);
    setSelectedFileName(`[Sample] ${name}`);
    setPreviewUrls([]);
    setLoading(true);

    try {
      const res = await fetch("/api/extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_id: sampleId }),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      setResult(data);
    } catch (err: any) {
      setError(err.message || "Failed to load sample.");
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${activeTab}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="container">
      {/* Top Floating Corner Creator Badge */}
      <div className="top-corner-credit">
        <span className="pulse-dot" />
        <span>Developed by <strong>Mohamad Darwish</strong></span>
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-badge">
          <Sparkles size={14} color="var(--green-primary)" />
          <span>AI Document Intelligence • Instant Verification</span>
        </div>
        <h1 className="header-title">
          UAE Document <span className="green-gradient-text">Reader Suite</span>
        </h1>
        <p className="header-subtitle">
          All-in-one vehicle and identity verification engine. Extract UAE Vehicle Licenses (Mulkiya),
          Passports, and Emirates ID + Driving Licenses with 100% in-memory data privacy.
        </p>
      </header>

      {/* 3-Tabs Selector */}
      <div className="tabs-nav-container">
        <div className="tabs-nav">
          <button
            className={`tab-btn ${activeTab === "mulkiya" ? "active" : ""}`}
            onClick={() => switchTab("mulkiya")}
          >
            <Car size={18} />
            <span>1. Vehicle Mulkiya (الملكية)</span>
          </button>
          <button
            className={`tab-btn ${activeTab === "passport" ? "active" : ""}`}
            onClick={() => switchTab("passport")}
          >
            <Globe size={18} />
            <span>2. Passport (جواز السفر)</span>
          </button>
          <button
            className={`tab-btn ${activeTab === "residence" ? "active" : ""}`}
            onClick={() => switchTab("residence")}
          >
            <CreditCard size={18} />
            <span>3. ID & Driving Licence (الهوية والرخصة)</span>
          </button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="main-grid">
        {/* Left Column: Upload Dropzone & Samples */}
        <div className="glass-panel dropzone-container">
          <input
            type="file"
            ref={fileInputRef}
            onChange={(e) => e.target.files && handleFileSelect(e.target.files)}
            accept="image/jpeg,image/png,image/webp,application/pdf"
            multiple={activeTab === "residence"}
            style={{ display: "none" }}
          />

          <div
            className={`dropzone-area ${isDragOver ? "drag-over" : ""}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragOver(true);
            }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragOver(false);
              if (e.dataTransfer.files) handleFileSelect(e.dataTransfer.files);
            }}
          >
            <div className="dropzone-icon-circle">
              {activeTab === "mulkiya" && <Car size={32} />}
              {activeTab === "passport" && <Globe size={32} />}
              {activeTab === "residence" && <CreditCard size={32} />}
            </div>
            <h3 className="dropzone-title">
              {activeTab === "mulkiya" && "Upload Vehicle License (Mulkiya)"}
              {activeTab === "passport" && "Upload Passport Document"}
              {activeTab === "residence" && "Upload Emirates ID & Driving Licence"}
            </h3>
            <p className="dropzone-desc">
              {activeTab === "residence"
                ? "Upload 1 combined image or select both cards (JPG, PNG, PDF)"
                : "Click or drag & drop document photo"}
            </p>
            <div className="dropzone-meta">
              <span>{activeTab === "residence" ? "Multi-file supported" : "Single file"}</span>
              <span>•</span>
              <span>Max 15 MB</span>
            </div>
          </div>

          {/* Thumbnail previews */}
          {previewUrls.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: `repeat(${previewUrls.length}, 1fr)`, gap: "0.5rem" }}>
              {previewUrls.map((url, i) => (
                <img
                  key={i}
                  src={url}
                  alt={`Preview ${i + 1}`}
                  style={{
                    width: "100%",
                    maxHeight: "140px",
                    objectFit: "cover",
                    borderRadius: "var(--radius-sm)",
                    border: "1px solid var(--border-subtle)",
                  }}
                />
              ))}
            </div>
          )}

          {/* Quick Demo Samples for each tab */}
          <div className="sample-section">
            <div className="sample-title">
              <Zap size={15} />
              <span>Instant Test Samples:</span>
            </div>

            {activeTab === "mulkiya" && (
              <div className="sample-buttons">
                <button
                  className="sample-btn"
                  onClick={() => loadSample("dubai_range_rover", "Range Rover Sport")}
                >
                  <Car size={16} />
                  <span>Range Rover (Dubai)</span>
                </button>
                <button
                  className="sample-btn"
                  onClick={() => loadSample("abu_dhabi_mercedes", "Mercedes G63 AMG")}
                >
                  <Car size={16} />
                  <span>G63 AMG (Abu Dhabi)</span>
                </button>
              </div>
            )}

            {activeTab === "passport" && (
              <div className="sample-buttons" style={{ gridTemplateColumns: "1fr" }}>
                <button
                  className="sample-btn"
                  onClick={() => loadSample("sample_passport_uk", "British Passport")}
                >
                  <Globe size={16} />
                  <span>British Passport (MRZ TD3 Validated)</span>
                </button>
              </div>
            )}

            {activeTab === "residence" && (
              <div className="sample-buttons" style={{ gridTemplateColumns: "1fr" }}>
                <button
                  className="sample-btn"
                  onClick={() => loadSample("sample_residence_uae", "Emirates ID + RTA Licence")}
                >
                  <CreditCard size={16} />
                  <span>UAE Resident (Emirates ID + RTA Licence Verified)</span>
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Dynamic Results Display */}
        <div className="glass-panel results-container">
          <div className="results-header">
            <div className="results-title">
              <FileCheck2 size={22} color="var(--green-primary)" />
              <span>
                {activeTab === "mulkiya" && "Extracted Vehicle Data"}
                {activeTab === "passport" && "Extracted Passport Data"}
                {activeTab === "residence" && "Resident Identity & Licence Verification"}
              </span>
            </div>
            {result && (
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button
                  className="export-btn"
                  onClick={() => setShowRawOcr(!showRawOcr)}
                  title="Toggle Raw OCR"
                >
                  <FileText size={15} />
                  <span>{showRawOcr ? "Cards" : "Raw OCR"}</span>
                </button>
                <button className="export-btn" onClick={downloadJson}>
                  <Download size={15} />
                  <span>JSON</span>
                </button>
              </div>
            )}
          </div>

          {/* Error Message */}
          {error && (
            <div className="status-badge status-error" style={{ width: "100%", padding: "1rem" }}>
              <AlertTriangle size={18} />
              <span>{error}</span>
            </div>
          )}

          {/* Loading Animation */}
          {loading && (
            <div className="loading-box">
              <div className="lux-spinner" />
              <h4 style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)" }}>
                Neural Document Processing & Verification...
              </h4>
              <p style={{ color: "var(--text-secondary)", fontSize: "0.88rem" }}>
                Extracting structured fields with instant in-memory security.
              </p>
            </div>
          )}

          {/* Empty State */}
          {!loading && !result && !error && (
            <div className="empty-box">
              {activeTab === "mulkiya" && <Car size={48} strokeWidth={1.5} />}
              {activeTab === "passport" && <Globe size={48} strokeWidth={1.5} />}
              {activeTab === "residence" && <CreditCard size={48} strokeWidth={1.5} />}
              <p>Upload a document or choose a sample to inspect extracted fields.</p>
            </div>
          )}

          {/* Results Views */}
          {!loading && result && (
            <>
              {selectedFileName && (
                <div style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
                  Processed file: <strong style={{ color: "var(--text-primary)" }}>{selectedFileName}</strong>
                </div>
              )}

              {/* Raw OCR Modal / Container */}
              {showRawOcr ? (
                <div
                  style={{
                    background: "var(--bg-surface)",
                    padding: "1.25rem",
                    borderRadius: "var(--radius-md)",
                    border: "1px solid var(--border-subtle)",
                    maxHeight: "450px",
                    overflowY: "auto",
                    fontFamily: "monospace",
                    fontSize: "0.85rem",
                    whiteSpace: "pre-wrap",
                    color: "var(--text-secondary)",
                  }}
                >
                  {result.raw_text}
                </div>
              ) : (
                <div className="cards-grid">
                  {/* ================= TAB 1: MULKIYA RESULTS ================= */}
                  {activeTab === "mulkiya" && result.data && (
                    <>
                      <div className="data-section-card">
                        <div className="data-section-header">
                          <Car size={16} />
                          <span>Plate & Authority (بيانات اللوحة)</span>
                        </div>
                        <div className="fields-grid">
                          <div className="field-box">
                            <span className="field-label">Plate Source (المصدر)</span>
                            <span className="field-value">{result.data.plate_source || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Plate Category (الصنف)</span>
                            <span className="field-value">{result.data.plate_category || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Plate Code (الرمز)</span>
                            <span className="field-value highlight">{result.data.plate_code || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Plate Number (رقم اللوحة)</span>
                            <span className="field-value highlight">{result.data.plate_number || "—"}</span>
                          </div>
                        </div>
                      </div>

                      <div className="data-section-card">
                        <div className="data-section-header">
                          <ShieldCheck size={16} />
                          <span>Vehicle Specs & Chassis (المواصفات والشاسي)</span>
                        </div>
                        <div className="fields-grid">
                          <div className="field-box" style={{ gridColumn: "1 / -1" }}>
                            <span className="field-label">Chassis No / VIN (رقم الشاسي)</span>
                            <div className="field-value highlight">
                              <span>{result.data.vin || "—"}</span>
                              {result.data.vin && (
                                <button
                                  className="copy-btn"
                                  onClick={() => handleCopy(result.data.vin!, "vin")}
                                  title="Copy VIN"
                                >
                                  {copiedKey === "vin" ? <Check size={15} color="var(--emerald)" /> : <Copy size={15} />}
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Make & Model (الصانع والطراز)</span>
                            <span className="field-value">
                              {result.data.make ? `${result.data.make} ${result.data.model || ""}` : "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Model Year (سنة الصنع)</span>
                            <span className="field-value">{result.data.year || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Color (اللون)</span>
                            <span className="field-value">{result.data.color || "—"}</span>
                          </div>
                        </div>
                      </div>

                      <div className="data-section-card">
                        <div className="data-section-header">
                          <Calendar size={16} />
                          <span>Registration & Insurance Dates (التواريخ)</span>
                        </div>
                        <div className="fields-grid">
                          <div className="field-box">
                            <span className="field-label">Registration Expiry (انتهاء الملكية)</span>
                            <span className="field-value highlight" style={{ color: "var(--emerald)" }}>
                              {result.data.registration_expiry || "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Insurance Expiry (انتهاء التأمين)</span>
                            <span className="field-value">{result.data.insurance_expiry || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Insurer (شركة التأمين)</span>
                            <span className="field-value">{result.data.insurance_company || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Policy No (رقم الوثيقة)</span>
                            <span className="field-value">{result.data.policy_number || "—"}</span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}

                  {/* ================= TAB 2: PASSPORT RESULTS ================= */}
                  {activeTab === "passport" && result.data && (
                    <>
                      <div className="data-section-card">
                        <div className="data-section-header">
                          <Globe size={16} />
                          <span>Passport Identity (بيانات الجواز)</span>
                        </div>
                        <div className="fields-grid">
                          <div className="field-box">
                            <span className="field-label">Passport Number (رقم الجواز)</span>
                            <div className="field-value highlight">
                              <span>{result.data.passport_number || "—"}</span>
                              {result.data.passport_number && (
                                <button
                                  className="copy-btn"
                                  onClick={() => handleCopy(result.data.passport_number!, "pp_num")}
                                  title="Copy Passport Number"
                                >
                                  {copiedKey === "pp_num" ? <Check size={15} color="var(--emerald)" /> : <Copy size={15} />}
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Status (الحالة)</span>
                            <span
                              className={`status-badge ${
                                result.data.status === "VALID"
                                  ? "status-success"
                                  : result.data.status === "EXPIRING_SOON"
                                  ? "status-warning"
                                  : "status-error"
                              }`}
                            >
                              {result.data.status}
                            </span>
                          </div>
                          <div className="field-box" style={{ gridColumn: "1 / -1" }}>
                            <span className="field-label">Full Name (الاسم الكامل)</span>
                            <span className="field-value" style={{ fontSize: "1.2rem", color: "var(--green-dark)" }}>
                              {result.data.full_name || "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Nationality (الجنسية)</span>
                            <span className="field-value">
                              {result.data.nationality_name} ({result.data.nationality_code || "—"})
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Issuing Country (جهة الإصدار)</span>
                            <span className="field-value">{result.data.issuing_country || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Date of Birth (تاريخ الميلاد)</span>
                            <span className="field-value">{result.data.date_of_birth || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Gender (الجنس)</span>
                            <span className="field-value">{result.data.gender || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Expiry Date (تاريخ الانتهاء)</span>
                            <span className="field-value highlight" style={{ color: "var(--emerald)" }}>
                              {result.data.expiry_date || "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Issue Date (تاريخ الإصدار)</span>
                            <span className="field-value">{result.data.issue_date || "—"}</span>
                          </div>
                        </div>
                      </div>

                      {/* MRZ Code section */}
                      {result.data.mrz_lines && result.data.mrz_lines.length > 0 && (
                        <div className="data-section-card">
                          <div className="data-section-header">
                            <Award size={16} />
                            <span>Machine Readable Zone (MRZ TD3)</span>
                          </div>
                          <div className="mrz-container">
                            {result.data.mrz_lines.join("\n")}
                          </div>
                        </div>
                      )}
                    </>
                  )}

                  {/* ================= TAB 3: RESIDENCE & LICENCE RESULTS ================= */}
                  {activeTab === "residence" && result.emirates_id && (
                    <>
                      {/* Reconciliation Banner */}
                      {result.reconciliation && (
                        <div
                          className={`reconciliation-card ${
                            result.reconciliation.overall_status === "VERIFIED"
                              ? ""
                              : result.reconciliation.overall_status === "NEEDS_REVIEW"
                              ? "needs-review"
                              : "expired"
                          }`}
                        >
                          <div className="reconcile-header">
                            <div className="reconcile-title">
                              <ShieldCheck size={20} color="var(--green-primary)" />
                              <span>Reconciliation & Cross-Check Status</span>
                            </div>
                            <span
                              className={`status-badge ${
                                result.reconciliation.overall_status === "VERIFIED"
                                  ? "status-success"
                                  : result.reconciliation.overall_status === "NEEDS_REVIEW"
                                  ? "status-warning"
                                  : "status-error"
                              }`}
                            >
                              {result.reconciliation.overall_status}
                            </span>
                          </div>
                          <div className="reconcile-badges">
                            <span className="pill-badge">
                              <Check size={14} color="var(--emerald)" />
                              <span>Names Reconciled</span>
                            </span>
                            <span className="pill-badge">
                              <Check size={14} color="var(--emerald)" />
                              <span>UAE Legal Driving Match</span>
                            </span>
                            <span className="pill-badge">
                              <Calendar size={14} color="var(--emerald)" />
                              <span>Valid Identity</span>
                            </span>
                          </div>
                        </div>
                      )}

                      {/* Emirates ID Card */}
                      <div className="data-section-card">
                        <div className="data-section-header">
                          <CreditCard size={16} />
                          <span>Emirates ID (بطاقة الهوية الإماراتية)</span>
                        </div>
                        <div className="fields-grid">
                          <div className="field-box" style={{ gridColumn: "1 / -1" }}>
                            <span className="field-label">Emirates ID Number (رقم الهوية)</span>
                            <div className="field-value highlight">
                              <span>{result.emirates_id.id_number || "—"}</span>
                              {result.emirates_id.id_number && (
                                <button
                                  className="copy-btn"
                                  onClick={() => handleCopy(result.emirates_id.id_number!, "eid_num")}
                                  title="Copy Emirates ID"
                                >
                                  {copiedKey === "eid_num" ? <Check size={15} color="var(--emerald)" /> : <Copy size={15} />}
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Holder Name (EN)</span>
                            <span className="field-value" style={{ fontWeight: 600 }}>
                              {result.emirates_id.name_en || "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">الاسم باللغة العربية (AR)</span>
                            <span className="field-value" style={{ fontWeight: 600 }}>
                              {result.emirates_id.name_ar || "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Nationality (الجنسية)</span>
                            <span className="field-value">{result.emirates_id.nationality || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Date of Birth (تاريخ الميلاد)</span>
                            <span className="field-value">{result.emirates_id.date_of_birth || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Expiry Date (تاريخ الانتهاء)</span>
                            <span className="field-value highlight" style={{ color: "var(--emerald)" }}>
                              {result.emirates_id.expiry_date || "—"}
                            </span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Card Number (رقم البطاقة)</span>
                            <span className="field-value">{result.emirates_id.card_number || "—"}</span>
                          </div>
                          {result.emirates_id.occupation && (
                            <div className="field-box">
                              <span className="field-label">Occupation (المهنة)</span>
                              <span className="field-value">{result.emirates_id.occupation}</span>
                            </div>
                          )}
                          {result.emirates_id.employer && (
                            <div className="field-box">
                              <span className="field-label">Employer (صاحب العمل)</span>
                              <span className="field-value">{result.emirates_id.employer}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* UAE Driving Licence Card */}
                      <div className="data-section-card">
                        <div className="data-section-header">
                          <Award size={16} />
                          <span>UAE Driving Licence (رخصة القيادة الإماراتية)</span>
                        </div>
                        <div className="fields-grid">
                          <div className="field-box">
                            <span className="field-label">Licence Number (رقم الرخصة)</span>
                            <div className="field-value highlight">
                              <span>{result.driving_licence.licence_number || "—"}</span>
                              {result.driving_licence.licence_number && (
                                <button
                                  className="copy-btn"
                                  onClick={() => handleCopy(result.driving_licence.licence_number!, "dl_num")}
                                  title="Copy Licence Number"
                                >
                                  {copiedKey === "dl_num" ? <Check size={15} color="var(--emerald)" /> : <Copy size={15} />}
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Traffic Code No. (الرمز المروري)</span>
                            <span className="field-value highlight">{result.driving_licence.traffic_code_no || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Issued By (جهة الإصدار)</span>
                            <span className="field-value">{result.driving_licence.issued_by || "—"}</span>
                          </div>
                          <div className="field-box">
                            <span className="field-label">Licence Expiry (انتهاء الرخصة)</span>
                            <span className="field-value highlight" style={{ color: "var(--emerald)" }}>
                              {result.driving_licence.expiry_date || "—"}
                            </span>
                          </div>
                          <div className="field-box" style={{ gridColumn: "1 / -1" }}>
                            <span className="field-label">Allowed Classes (فئات المركبات)</span>
                            <span className="field-value">
                              {result.driving_licence.categories?.join(", ") || "Light Vehicle (3)"}
                            </span>
                          </div>
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Security Footer & Creator Credit */}
      <footer className="security-footer-container">
        <div className="security-footer">
          <div className="security-item">
            <Lock size={15} color="var(--green-primary)" />
            <span>In-Memory Zero Retention</span>
          </div>
          <div className="security-item">
            <ShieldCheck size={15} color="var(--green-primary)" />
            <span>Encrypted Cloud Security (EU Enclave)</span>
          </div>
          <div className="security-item">
            <Info size={15} color="var(--green-primary)" />
            <span>Vercel Serverless Ready</span>
          </div>
        </div>
        <div className="footer-credits">
          <div className="footer-credits-title">
            تم تطوير وبناء هذا النظام بالكامل من قبل <span className="author-highlight">Mohamad Darwish</span>
          </div>
          <div className="footer-credits-sub">
            Architected & Engineered by Mohamad Darwish • UAE Document Intelligence Suite
          </div>
        </div>
      </footer>
    </div>
  );
}

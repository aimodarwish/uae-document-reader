import { NextRequest, NextResponse } from "next/server";
import { processDocumentWithDocAI } from "@/lib/documentai";
import { extractMulkiyaFields } from "@/lib/mulkiya-extractor";
import { extractPassportFields } from "@/lib/passport-extractor";
import { extractResidenceFields } from "@/lib/residence-extractor";
import {
  SAMPLE_MULKIYA_RESULTS,
  SAMPLE_PASSPORT_RESULTS,
  SAMPLE_RESIDENCE_RESULTS,
} from "@/lib/sample-data";

export const maxDuration = 60; // 60 seconds timeout for Vercel functions
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  const startTime = Date.now();

  try {
    const contentType = req.headers.get("content-type") || "";

    // 1. Handle JSON request (Samples or direct Base64)
    if (contentType.includes("application/json")) {
      const body = await req.json();

      // Sample requests
      if (body.sample_id) {
        if (SAMPLE_MULKIYA_RESULTS[body.sample_id]) {
          return NextResponse.json({
            ...SAMPLE_MULKIYA_RESULTS[body.sample_id],
            doc_type: "mulkiya",
            is_sample: true,
            processing_time_ms: 110,
          });
        }
        if (SAMPLE_PASSPORT_RESULTS[body.sample_id]) {
          return NextResponse.json({
            ...SAMPLE_PASSPORT_RESULTS[body.sample_id],
            doc_type: "passport",
            is_sample: true,
            processing_time_ms: 95,
          });
        }
        if (SAMPLE_RESIDENCE_RESULTS[body.sample_id]) {
          return NextResponse.json({
            ...SAMPLE_RESIDENCE_RESULTS[body.sample_id],
            doc_type: "residence",
            is_sample: true,
            processing_time_ms: 130,
          });
        }
      }

      if (body.base64) {
        const mimeType = body.mimeType || "image/jpeg";
        const buffer = Buffer.from(body.base64, "base64");
        const docAiRes = await processDocumentWithDocAI(buffer, mimeType);
        const docType = body.doc_type || "mulkiya";

        let extractionResult: any;
        if (docType === "passport") {
          extractionResult = extractPassportFields(docAiRes);
        } else if (docType === "residence") {
          extractionResult = extractResidenceFields(docAiRes);
        } else {
          extractionResult = extractMulkiyaFields(docAiRes);
        }

        return NextResponse.json({
          ...extractionResult,
          doc_type: docType,
          is_sample: false,
          processing_time_ms: Date.now() - startTime,
        });
      }
    }

    // 2. Handle Multipart / FormData upload
    const formData = await req.formData();
    const docType = (formData.get("doc_type") as string) || "mulkiya";

    // Collect uploaded files (supports single "file" or multiple "files")
    const files: File[] = [];
    const singleFile = formData.get("file") as File | null;
    if (singleFile) files.push(singleFile);

    const allFiles = formData.getAll("files") as File[];
    for (const f of allFiles) {
      if (f && !files.includes(f)) files.push(f);
    }

    if (files.length === 0) {
      return NextResponse.json(
        { success: false, error: "No document file was uploaded." },
        { status: 400 }
      );
    }

    // Process all uploaded files concurrently with Promise.all
    const docAiResults = await Promise.all(
      files.map(async (f) => {
        const buf = Buffer.from(await f.arrayBuffer());
        return processDocumentWithDocAI(buf, f.type || "image/jpeg");
      })
    );

    // Merge OCR text and lines across all uploaded images/pages
    const mergedDocAiRes = {
      rawText: docAiResults.map((r, i) => `--- PAGE ${i + 1} (${files[i]?.name || "Doc"}) ---\n` + r.rawText).join("\n\n"),
      lines: docAiResults.flatMap((r) => r.lines),
      pageCount: docAiResults.reduce((acc, r) => acc + r.pageCount, 0),
      dimensions: docAiResults[0]?.dimensions || { width: 1000, height: 1000 },
    };

    let extractionResult: any;
    if (docType === "passport") {
      extractionResult = extractPassportFields(mergedDocAiRes);
    } else if (docType === "residence") {
      extractionResult = extractResidenceFields(mergedDocAiRes);
    } else {
      extractionResult = extractMulkiyaFields(mergedDocAiRes);
    }

    return NextResponse.json({
      ...extractionResult,
      doc_type: docType,
      filename: files.map((f) => f.name).join(" + "),
      is_sample: false,
      processing_time_ms: Date.now() - startTime,
    });
  } catch (error: any) {
    console.error("Extraction error:", error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || "An unexpected error occurred during document extraction.",
        processing_time_ms: Date.now() - startTime,
      },
      { status: 500 }
    );
  }
}

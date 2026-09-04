import crypto from "node:crypto";

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ExtractedLine {
  text: string;
  confidence: number;
  box: BoundingBox;
  page: number;
  lang?: string;
}

export interface DocumentAIResponse {
  rawText: string;
  lines: ExtractedLine[];
  pageCount: number;
  dimensions: { width: number; height: number };
}

let cachedToken: { token: string; expiresAt: number } | null = null;

function b64url(input: string | Buffer): string {
  const buf = typeof input === "string" ? Buffer.from(input, "utf8") : input;
  return buf.toString("base64url");
}

/**
 * Generates an OAuth2 Access Token for Google Cloud using Service Account credentials.
 * Zero external libraries needed - uses native node:crypto.
 */
export async function getGoogleAccessToken(): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  if (cachedToken && cachedToken.expiresAt > now + 60) {
    return cachedToken.token;
  }

  const clientEmail = process.env.GCP_CLIENT_EMAIL;
  let privateKey = process.env.GCP_PRIVATE_KEY;

  if (!clientEmail || !privateKey) {
    throw new Error("Missing GCP_CLIENT_EMAIL or GCP_PRIVATE_KEY environment variables.");
  }

  // Replace literal '\n' strings with real newlines if passed in .env
  privateKey = privateKey.replace(/\\n/g, "\n");

  const header = { alg: "RS256", typ: "JWT" };
  const payload = {
    iss: clientEmail,
    scope: "https://www.googleapis.com/auth/cloud-platform",
    aud: "https://oauth2.googleapis.com/token",
    exp: now + 3600,
    iat: now,
  };

  const headerB64 = b64url(JSON.stringify(header));
  const payloadB64 = b64url(JSON.stringify(payload));
  const unsignedJwt = `${headerB64}.${payloadB64}`;

  const signer = crypto.createSign("RSA-SHA256");
  signer.update(unsignedJwt);
  const signature = signer.sign(privateKey, "base64url");
  const jwt = `${unsignedJwt}.${signature}`;

  const res = await fetch("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion=${jwt}`,
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Google OAuth token request failed (${res.status}): ${errorText}`);
  }

  const data = (await res.json()) as { access_token: string; expires_in: number };
  cachedToken = {
    token: data.access_token,
    expiresAt: now + (data.expires_in || 3600),
  };

  return data.access_token;
}

/**
 * Calls Google Cloud Document AI to process an image or PDF.
 */
export async function processDocumentWithDocAI(
  fileBuffer: Buffer,
  mimeType: string
): Promise<DocumentAIResponse> {
  const token = await getGoogleAccessToken();

  const location = process.env.GCP_LOCATION || "eu";
  const processorId = process.env.GCP_PROCESSOR_ID || "33d1dea3952e2d2c";
  const projectNumber = process.env.GCP_PROJECT_NUMBER || "976524610604";

  const endpoint = `https://${location}-documentai.googleapis.com/v1/projects/${projectNumber}/locations/${location}/processors/${processorId}:process`;

  const requestBody = {
    rawDocument: {
      content: fileBuffer.toString("base64"),
      mimeType: mimeType || "image/jpeg",
    },
  };

  const res = await fetch(endpoint, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Document AI API error (${res.status}): ${errText}`);
  }

  const result = await res.json();
  const doc = result.document;
  if (!doc) {
    throw new Error("Document AI returned an empty document object.");
  }

  const fullText: string = doc.text || "";
  const lines: ExtractedLine[] = [];
  let width = 1000;
  let height = 1000;

  if (doc.pages && doc.pages.length > 0) {
    for (const page of doc.pages) {
      const pNum = page.pageNumber || 1;
      if (page.dimension) {
        width = page.dimension.width || width;
        height = page.dimension.height || height;
      }

      // Process lines
      if (page.lines) {
        for (const line of page.lines) {
          let lineText = "";
          if (line.layout?.textAnchor?.textSegments) {
            for (const seg of line.layout.textAnchor.textSegments) {
              const start = parseInt(seg.startIndex || "0", 10);
              const end = parseInt(seg.endIndex || "0", 10);
              lineText += fullText.substring(start, end);
            }
          }

          const vertices = line.layout?.boundingPoly?.normalizedVertices || [];
          let x1 = 0, y1 = 0, x2 = 1, y2 = 1;
          if (vertices.length >= 2) {
            x1 = Math.min(...vertices.map((v: any) => v.x ?? 0));
            y1 = Math.min(...vertices.map((v: any) => v.y ?? 0));
            x2 = Math.max(...vertices.map((v: any) => v.x ?? 0));
            y2 = Math.max(...vertices.map((v: any) => v.y ?? 0));
          }

          const confidence = line.layout?.confidence ?? 0.9;
          const cleanedText = lineText.replace(/\r?\n|\r/g, " ").trim();

          if (cleanedText) {
            lines.push({
              text: cleanedText,
              confidence,
              box: {
                x1: Math.round(x1 * width),
                y1: Math.round(y1 * height),
                x2: Math.round(x2 * width),
                y2: Math.round(y2 * height),
              },
              page: pNum,
            });
          }
        }
      }
    }
  }

  return {
    rawText: fullText,
    lines,
    pageCount: doc.pages?.length || 1,
    dimensions: { width, height },
  };
}

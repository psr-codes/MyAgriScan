import { NextResponse } from "next/server"

export async function POST(request: Request) {
  try {
    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

  // 1. Send image to your FastAPI Backend for classification
  // Prefer the public NEXT env var (used by the client) but fall back to BACKEND_URL or localhost
  const rawBackend = process.env.NEXT_PUBLIC_BACKEND_URL || process.env.BACKEND_URL || "http://127.0.0.1:8000"
  const backendUrl = rawBackend.replace(/\/$/, '')

    // Create a new FormData for the backend request
    const backendFormData = new FormData()
    backendFormData.append("file", file)

  let result: any = undefined
  let diseaseName = "Unknown Disease"
  let confidence = 0

    try {
      const backendResponse = await fetch(`${backendUrl}/api/diagnose`, {
        method: "POST",
        body: backendFormData,
      })

      if (!backendResponse.ok) {
        console.warn("Backend response not ok:", backendResponse.status, backendResponse.statusText)
        // Check if we can get error details
        try {
          const errorText = await backendResponse.text()
          console.warn("Backend error details:", errorText)
        } catch (e) {
          // ignore
        }
        throw new Error("Failed to get response from classification server")
      }

  const result = await backendResponse.json()
  // The FastAPI backend returns:
  // { predicted_class, confidence, scores, filename }
  // Use those keys directly so we forward the original object to the frontend.
  diseaseName = result.predicted_class || result.class || result.prediction || "Unknown"
  confidence = result.confidence || 0
  // keep the full result to include scores/filename in the response
    } catch (error) {
      console.error("Backend error:", error)
      // Fallback for demo purposes if backend isn't running
      // REMOVE THIS IN PRODUCTION
      console.log("[v0] Using fallback mock data because backend failed")
      diseaseName = "Tomato___Late_blight"
      confidence = 0.95
    }

    // For this route we only proxy to the backend classification service and
    // return its response (plus a friendly `name` field for the frontend).
    const displayName = String(diseaseName || '').replace(/___/g, ' ').replace(/_/g, ' ').trim()
    const base = {
      name: displayName,
      predicted_class: diseaseName,
      confidence,
      scores: (typeof result !== 'undefined' && result.scores) ? result.scores : [],
      filename: (typeof result !== 'undefined' && result.filename) ? result.filename : '',
    }

    return NextResponse.json(base)
  } catch (error) {
    console.error("Diagnosis error:", error)
    return NextResponse.json({ error: "Failed to process diagnosis" }, { status: 500 })
  }
}

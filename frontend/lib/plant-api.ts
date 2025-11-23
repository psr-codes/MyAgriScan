import type { DiseaseInfo } from "./types"

// This mocks the backend response from your FastAPI + Gemini integration
export async function analyzeImage(file: File): Promise<DiseaseInfo> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch("/api/diagnose", {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}))
    console.error("API Error:", errorData)
    throw new Error(errorData.error || "Diagnosis failed")
  }

  const data = await response.json()
  console.log("analyzeImage response:", data)

  // Normalize confidence: backend may return percentage (0-100) or fraction (0-1)
  const rawConfidence = typeof data.confidence === 'number' ? data.confidence : 0
  const confidence = rawConfidence > 1 ? rawConfidence / 100 : rawConfidence

  // Fetch description from server-side Gemini route
  try {
    const descRes = await fetch('/api/get-description', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ predicted_class: data.predicted_class || data.predictedClass || data.class }),
    })
    const descJson = await descRes.json()
    console.log('get-description response:', descJson)

    return {
      name: data.name || (data.predicted_class || data.predictedClass || '').replace(/___/g, ' ').replace(/_/g, ' '),
      confidence,
      description: descJson.description || '',
      prevention: descJson.prevention || [],
      treatment: descJson.treatment || [],
      tips: descJson.tips || [],
    }
  } catch (e) {
    console.warn('Failed to fetch description:', e)
    return {
      name: data.name || (data.predicted_class || data.predictedClass || '').replace(/___/g, ' ').replace(/_/g, ' '),
      confidence,
      description: '',
      prevention: [],
      treatment: [],
      tips: [],
    }
  }
}

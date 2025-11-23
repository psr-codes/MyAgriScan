import { NextResponse } from 'next/server'

type RequestBody = {
  predicted_class?: string
  name?: string
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as RequestBody
    const predicted = body.predicted_class || body.name || ''
    if (!predicted) return NextResponse.json({ error: 'predicted_class required' }, { status: 400 })

    const apiKey = process.env.NEXT_PUBLIC_GEMINI_API_KEY
    if (!apiKey) {
      return NextResponse.json({ gemError: { status: 401, detail: 'Gemini API key not configured on server' } }, { status: 200 })
    }

    const rawModel = process.env.GEMINI_MODEL || 'gemini-2.0-flash'
    const modelSegment = rawModel.includes('/') ? rawModel : `models/${rawModel}`
    const url = `https://generativelanguage.googleapis.com/v1beta/${modelSegment}:generateContent`

    // Prompt asks for compact JSON only to make parsing reliable
    const prompt = `The plant disease detected is: "${predicted}". Return ONLY a single, compact JSON object (no markdown or code fences) with keys: description (2-3 sentences), prevention (array of short strings), treatment (array of short strings), tips (array of short strings). Keep answers short and farmer-friendly.`

    const payload = { contents: [{ parts: [{ text: prompt }] }] }

    try {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-goog-api-key': apiKey },
        body: JSON.stringify(payload),
      })

      if (!resp.ok) {
        let bodyText = ''
        try { bodyText = JSON.stringify(await resp.json()) } catch { bodyText = await resp.text() }
        return NextResponse.json({ gemError: { status: resp.status, detail: bodyText } }, { status: 200 })
      }

      const data = await resp.json()
      // extract candidate text
      let fullText = ''
      try {
        const cand = data?.candidates?.[0]
        if (cand) {
          if (cand.content && cand.content.parts && Array.isArray(cand.content.parts)) {
            fullText = cand.content.parts.map((p: any) => p.text || '').join('\n')
          } else if (Array.isArray(cand.content)) {
            const pieces: string[] = []
            for (const item of cand.content) {
              if (item.parts && Array.isArray(item.parts)) pieces.push(...item.parts.map((pp: any) => pp.text || ''))
              else if (typeof item.text === 'string') pieces.push(item.text)
            }
            fullText = pieces.join('\n')
          } else if (typeof cand.output === 'string') {
            fullText = cand.output
          }
        }
        if (!fullText && typeof data?.text === 'string') fullText = data.text
      } catch (e) {
        console.error('Error extracting Gemini candidate text', e)
      }

      const stripped = (fullText || '').replace(/```json/g, '').replace(/```/g, '').trim()
      let parsed: any = null
      try {
        parsed = JSON.parse(stripped)
      } catch (e) {
        // fallback: try to find first JSON object inside the text
        const m = stripped.match(/\{[\s\S]*\}/)
        if (m) {
          try { parsed = JSON.parse(m[0]) } catch { parsed = null }
        }
      }

      function toArray(v: any): string[] {
        if (!v && v !== 0) return []
        if (Array.isArray(v)) return v.map((x) => String(x))
        if (typeof v === 'string') return v.split(/\r?\n|;|\.|\u2022/).map(s => s.trim()).filter(Boolean)
        return [String(v)]
      }

      if (!parsed) {
        return NextResponse.json({ description: stripped.substring(0, 400), prevention: [], treatment: [], tips: [], raw: fullText, gemError: { status: 2, detail: 'Failed to parse JSON from Gemini' } }, { status: 200 })
      }

      const out = {
        description: String(parsed.description || parsed.desc || parsed.summary || ''),
        prevention: toArray(parsed.prevention || parsed.prevent || parsed.preventions || []),
        treatment: toArray(parsed.treatment || parsed.treatments || parsed.remedy || []),
        tips: toArray(parsed.tips || parsed.tip || parsed.advice || []),
        raw: fullText || '',
      }

      return NextResponse.json(out)
    } catch (e: any) {
      console.error('Gemini request failed', e)
      return NextResponse.json({ gemError: { status: 0, detail: String(e?.message || e) } }, { status: 200 })
    }
  } catch (err: any) {
    console.error('get-description error', err)
    return NextResponse.json({ error: String(err?.message || err) }, { status: 500 })
  }
}

module.exports = [
"[externals]/next/dist/compiled/next-server/app-route-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-route-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-route-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/@opentelemetry/api [external] (next/dist/compiled/@opentelemetry/api, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/@opentelemetry/api", () => require("next/dist/compiled/@opentelemetry/api"));

module.exports = mod;
}),
"[externals]/next/dist/compiled/next-server/app-page-turbo.runtime.dev.js [external] (next/dist/compiled/next-server/app-page-turbo.runtime.dev.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js", () => require("next/dist/compiled/next-server/app-page-turbo.runtime.dev.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-unit-async-storage.external.js [external] (next/dist/server/app-render/work-unit-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-unit-async-storage.external.js", () => require("next/dist/server/app-render/work-unit-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/work-async-storage.external.js [external] (next/dist/server/app-render/work-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/work-async-storage.external.js", () => require("next/dist/server/app-render/work-async-storage.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/shared/lib/no-fallback-error.external.js [external] (next/dist/shared/lib/no-fallback-error.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/shared/lib/no-fallback-error.external.js", () => require("next/dist/shared/lib/no-fallback-error.external.js"));

module.exports = mod;
}),
"[externals]/next/dist/server/app-render/after-task-async-storage.external.js [external] (next/dist/server/app-render/after-task-async-storage.external.js, cjs)", ((__turbopack_context__, module, exports) => {

const mod = __turbopack_context__.x("next/dist/server/app-render/after-task-async-storage.external.js", () => require("next/dist/server/app-render/after-task-async-storage.external.js"));

module.exports = mod;
}),
"[project]/app/api/get-description/route.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "POST",
    ()=>POST
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/server.js [app-route] (ecmascript)");
;
async function POST(request) {
    try {
        const body = await request.json();
        const predicted = body.predicted_class || body.name || '';
        if (!predicted) return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: 'predicted_class required'
        }, {
            status: 400
        });
        const apiKey = ("TURBOPACK compile-time value", "AIzaSyBKHhJTfOloGTVOR1UaWZtFvZ6A7qhXl-A");
        if ("TURBOPACK compile-time falsy", 0) //TURBOPACK unreachable
        ;
        const rawModel = process.env.GEMINI_MODEL || 'gemini-2.0-flash';
        const modelSegment = rawModel.includes('/') ? rawModel : `models/${rawModel}`;
        const url = `https://generativelanguage.googleapis.com/v1beta/${modelSegment}:generateContent`;
        // Prompt asks for compact JSON only to make parsing reliable
        const prompt = `The plant disease detected is: "${predicted}". Return ONLY a single, compact JSON object (no markdown or code fences) with keys: description (2-3 sentences), prevention (array of short strings), treatment (array of short strings), tips (array of short strings). Keep answers short and farmer-friendly.`;
        const payload = {
            contents: [
                {
                    parts: [
                        {
                            text: prompt
                        }
                    ]
                }
            ]
        };
        try {
            const resp = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-goog-api-key': apiKey
                },
                body: JSON.stringify(payload)
            });
            if (!resp.ok) {
                let bodyText = '';
                try {
                    bodyText = JSON.stringify(await resp.json());
                } catch  {
                    bodyText = await resp.text();
                }
                return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
                    gemError: {
                        status: resp.status,
                        detail: bodyText
                    }
                }, {
                    status: 200
                });
            }
            const data = await resp.json();
            // extract candidate text
            let fullText = '';
            try {
                const cand = data?.candidates?.[0];
                if (cand) {
                    if (cand.content && cand.content.parts && Array.isArray(cand.content.parts)) {
                        fullText = cand.content.parts.map((p)=>p.text || '').join('\n');
                    } else if (Array.isArray(cand.content)) {
                        const pieces = [];
                        for (const item of cand.content){
                            if (item.parts && Array.isArray(item.parts)) pieces.push(...item.parts.map((pp)=>pp.text || ''));
                            else if (typeof item.text === 'string') pieces.push(item.text);
                        }
                        fullText = pieces.join('\n');
                    } else if (typeof cand.output === 'string') {
                        fullText = cand.output;
                    }
                }
                if (!fullText && typeof data?.text === 'string') fullText = data.text;
            } catch (e) {
                console.error('Error extracting Gemini candidate text', e);
            }
            const stripped = (fullText || '').replace(/```json/g, '').replace(/```/g, '').trim();
            let parsed = null;
            try {
                parsed = JSON.parse(stripped);
            } catch (e) {
                // fallback: try to find first JSON object inside the text
                const m = stripped.match(/\{[\s\S]*\}/);
                if (m) {
                    try {
                        parsed = JSON.parse(m[0]);
                    } catch  {
                        parsed = null;
                    }
                }
            }
            function toArray(v) {
                if (!v && v !== 0) return [];
                if (Array.isArray(v)) return v.map((x)=>String(x));
                if (typeof v === 'string') return v.split(/\r?\n|;|\.|\u2022/).map((s)=>s.trim()).filter(Boolean);
                return [
                    String(v)
                ];
            }
            if (!parsed) {
                return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
                    description: stripped.substring(0, 400),
                    prevention: [],
                    treatment: [],
                    tips: [],
                    raw: fullText,
                    gemError: {
                        status: 2,
                        detail: 'Failed to parse JSON from Gemini'
                    }
                }, {
                    status: 200
                });
            }
            const out = {
                description: String(parsed.description || parsed.desc || parsed.summary || ''),
                prevention: toArray(parsed.prevention || parsed.prevent || parsed.preventions || []),
                treatment: toArray(parsed.treatment || parsed.treatments || parsed.remedy || []),
                tips: toArray(parsed.tips || parsed.tip || parsed.advice || []),
                raw: fullText || ''
            };
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json(out);
        } catch (e) {
            console.error('Gemini request failed', e);
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
                gemError: {
                    status: 0,
                    detail: String(e?.message || e)
                }
            }, {
                status: 200
            });
        }
    } catch (err) {
        console.error('get-description error', err);
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: String(err?.message || err)
        }, {
            status: 500
        });
    }
}
}),
];

//# sourceMappingURL=%5Broot-of-the-server%5D__323daad9._.js.map
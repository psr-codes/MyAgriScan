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
"[project]/app/api/diagnose/route.ts [app-route] (ecmascript)", ((__turbopack_context__) => {
"use strict";

__turbopack_context__.s([
    "POST",
    ()=>POST
]);
var __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__ = __turbopack_context__.i("[project]/node_modules/next/server.js [app-route] (ecmascript)");
;
async function POST(request) {
    try {
        const formData = await request.formData();
        const file = formData.get("file");
        if (!file) {
            return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
                error: "No file provided"
            }, {
                status: 400
            });
        }
        // 1. Send image to your FastAPI Backend for classification
        // Prefer the public NEXT env var (used by the client) but fall back to BACKEND_URL or localhost
        const rawBackend = ("TURBOPACK compile-time value", "http://localhost:8000") || process.env.BACKEND_URL || "http://127.0.0.1:8000";
        const backendUrl = rawBackend.replace(/\/$/, '');
        // Create a new FormData for the backend request
        const backendFormData = new FormData();
        backendFormData.append("file", file);
        let result = undefined;
        let diseaseName = "Unknown Disease";
        let confidence = 0;
        try {
            const backendResponse = await fetch(`${backendUrl}/api/diagnose`, {
                method: "POST",
                body: backendFormData
            });
            if (!backendResponse.ok) {
                console.warn("Backend response not ok:", backendResponse.status, backendResponse.statusText);
                // Check if we can get error details
                try {
                    const errorText = await backendResponse.text();
                    console.warn("Backend error details:", errorText);
                } catch (e) {
                // ignore
                }
                throw new Error("Failed to get response from classification server");
            }
            const result = await backendResponse.json();
            // The FastAPI backend returns:
            // { predicted_class, confidence, scores, filename }
            // Use those keys directly so we forward the original object to the frontend.
            diseaseName = result.predicted_class || result.class || result.prediction || "Unknown";
            confidence = result.confidence || 0;
        // keep the full result to include scores/filename in the response
        } catch (error) {
            console.error("Backend error:", error);
            // Fallback for demo purposes if backend isn't running
            // REMOVE THIS IN PRODUCTION
            console.log("[v0] Using fallback mock data because backend failed");
            diseaseName = "Tomato___Late_blight";
            confidence = 0.95;
        }
        // For this route we only proxy to the backend classification service and
        // return its response (plus a friendly `name` field for the frontend).
        const displayName = String(diseaseName || '').replace(/___/g, ' ').replace(/_/g, ' ').trim();
        const base = {
            name: displayName,
            predicted_class: diseaseName,
            confidence,
            scores: typeof result !== 'undefined' && result.scores ? result.scores : [],
            filename: typeof result !== 'undefined' && result.filename ? result.filename : ''
        };
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json(base);
    } catch (error) {
        console.error("Diagnosis error:", error);
        return __TURBOPACK__imported__module__$5b$project$5d2f$node_modules$2f$next$2f$server$2e$js__$5b$app$2d$route$5d$__$28$ecmascript$29$__["NextResponse"].json({
            error: "Failed to process diagnosis"
        }, {
            status: 500
        });
    }
}
}),
];

//# sourceMappingURL=%5Broot-of-the-server%5D__a40841d0._.js.map
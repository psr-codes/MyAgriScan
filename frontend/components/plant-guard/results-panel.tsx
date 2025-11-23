"use client"

import type { DiseaseInfo } from "@/lib/types"
import { AlertTriangle, CheckCircle, Sprout, ShieldCheck, Info } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface ResultsPanelProps {
  disease: DiseaseInfo
  imageUrl: string
  onReset: () => void
}

export function ResultsPanel({ disease, imageUrl, onReset }: ResultsPanelProps) {
  const isHealthy = (disease?.name || '').toLowerCase().includes("healthy")

  return (
    <div className="w-full max-w-6xl mx-auto animate-in fade-in slide-in-from-bottom-4 duration-500 p-4">
      <div className="grid md:grid-cols-2 gap-8">
        {/* Left Column - Image & Diagnosis */}
        <div className="space-y-6">
          <Card className="overflow-hidden border-2 border-primary/20 shadow-lg h-full flex flex-col justify-between">
           <div>
             <div className="relative aspect-video w-full bg-muted">
              <img src={imageUrl || "/placeholder.svg"} alt="Analyzed Plant" className="w-full h-full object-cover" />
              <div className="absolute top-4 right-4">
                <Badge variant={isHealthy ? "default" : "destructive"} className="text-lg px-4 py-1 shadow-md">
                  {(disease.confidence * 100).toFixed(1)}% Match
                </Badge>
              </div>
            </div>

            <CardHeader
              className={isHealthy ? "bg-green-50 dark:bg-green-950/30" : "bg-orange-50 dark:bg-orange-950/30"}
            >
              <div className="flex items-center gap-3">
                {isHealthy ? (
                  <CheckCircle className="w-8 h-8 text-green-600" />
                ) : (
                  <AlertTriangle className="w-8 h-8 text-orange-600" />
                )}
                <div>
                  <CardTitle className="text-2xl font-bold">{disease.name}</CardTitle>
                  <p className="text-sm text-muted-foreground">AI Diagnosis Result</p>
                </div>
              </div>
            </CardHeader>

            <CardContent className="p-6">
              <p className="text-lg leading-relaxed text-muted-foreground">{disease.description}</p>
            </CardContent>
           </div>

              {/* Analyze another button moved here (below left card) */}
          <div className="flex justify-center border mx-auto p-3 rounded-md mt-4 hover:shadow-md transition-shadow cursor-pointer bg-white">
            <button onClick={onReset} className="text-primary font-medium  text-lg">
              Analyze Another Plant
            </button>
          </div>
          </Card>

        

        </div>

        {/* Right Column - Details & Recommendations */}
        <div className="space-y-6">
          <Card className="border-primary/20 shadow-md">
            <CardHeader className="bg-secondary/30 pb-3">
              <div className="flex items-center gap-3">
                <ShieldCheck className="w-6 h-6 text-primary" />
                <CardTitle className="text-xl">Treatment Options</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              { (disease.treatment && disease.treatment.length > 0) ? (
                <ul className="space-y-3">
                  {(disease.treatment || []).map((item, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <span className="flex-shrink-0 w-2 h-2 rounded-full bg-primary mt-2" />
                      <span className="text-base">{item}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-muted-foreground">No treatment suggestions available for this diagnosis.</p>
              )}
            </CardContent>
          </Card>

          <Card className="border-primary/20 shadow-md">
            <CardHeader className="bg-secondary/30 pb-3">
              <div className="flex items-center gap-3">
                <Sprout className="w-6 h-6 text-primary" />
                <CardTitle className="text-xl">Prevention & Care</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <ul className="space-y-3">
                {(disease.prevention || []).map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-2 h-2 rounded-full bg-primary mt-2" />
                    <span className="text-base">{item}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>

          <Card className="border-primary/20 shadow-md">
            <CardHeader className="bg-secondary/30 pb-3">
              <div className="flex items-center gap-3">
                <Info className="w-6 h-6 text-primary" />
                <CardTitle className="text-xl">Farming Tips</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="pt-6">
              <ul className="space-y-3">
                {(disease.tips || []).map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-2 h-2 rounded-full bg-primary mt-2" />
                    <span className="text-base">{item}</span>
                  </li>
                ))}
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>

      
    </div>
  )
}

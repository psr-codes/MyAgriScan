"use client"

import { useState, useEffect } from "react"
 import { UploadZone } from "@/components/plant-guard/upload-zone"
import { ResultsPanel } from "@/components/plant-guard/results-panel"
import { HistoryList } from "@/components/plant-guard/history-list"
import { analyzeImage } from "@/lib/plant-api"
import type { DiseaseInfo, ScanResult } from "@/lib/types"

export default function PlantDoctorPage() {
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [currentResult, setCurrentResult] = useState<DiseaseInfo | null>(null)
  const [currentImage, setCurrentImage] = useState<string | null>(null)
  const [history, setHistory] = useState<ScanResult[]>([])

  // Load history from local storage on mount
  useEffect(() => {
    const saved = localStorage.getItem("plant-guard-history")
    if (saved) {
      try {
        setHistory(JSON.parse(saved))
      } catch (e) {
        console.error("Failed to parse history", e)
      }
    }
  }, [])

  // Save history to local storage whenever it changes
  useEffect(() => {
    localStorage.setItem("plant-guard-history", JSON.stringify(history))
  }, [history])

  const handleImageSelect = async (file: File) => {
    setIsAnalyzing(true)

    // Create a local URL for the image preview
    const imageUrl = URL.createObjectURL(file)
    setCurrentImage(imageUrl)

    try {
      // Call our mock API (which simulates the backend + Gemini)
      const result = await analyzeImage(file)

      console.log("result is xxx:", result)
      setCurrentResult(result)

      // Add to history
      const newScan: ScanResult = {
        id: Date.now().toString(),
        date: new Date().toISOString(),
        imageUrl,
        disease: result,
      }

      setHistory((prev) => [newScan, ...prev].slice(0, 10)) // Keep last 10
    } catch (error) {
      console.error("Analysis failed:", error)
      // In a real app, show error toast
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setCurrentResult(null)
    setCurrentImage(null)
  }

  const handleHistorySelect = (scan: ScanResult) => {
    setCurrentImage(scan.imageUrl)
    setCurrentResult(scan.disease)
    window.scrollTo({ top: 0, behavior: "smooth" })
  }

  return (
    <div className="min-h-screen bg-background flex flex-col font-sans">
 
      <main className="flex-1 container max-w-4xl mx-auto px-4 py-8">
        {!currentResult ? (
          <div className="space-y-12 animate-in fade-in duration-500">
            <div className="text-center space-y-4 pt-8">
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Heal Your Crop, <br className="hidden sm:block" />
                <span className="text-primary">Harvest Hope.</span>
              </h2>
              <p className="text-lg text-muted-foreground max-w-lg mx-auto">
                Upload a photo of your plant leaf. Our AI will instantly detect diseases and provide expert treatment
                advice.
              </p>
            </div>

            <UploadZone onImageSelect={handleImageSelect} isAnalyzing={isAnalyzing} />

            <HistoryList history={history} onSelect={handleHistorySelect} />
          </div>
        ) : (
          <ResultsPanel disease={currentResult} imageUrl={currentImage || ""} onReset={handleReset} />
        )}
      </main>

      <footer className="py-8 bg-secondary/30 border-t border-border mt-auto">
        <div className="container mx-auto px-4 text-center">
          <p className="text-sm font-medium text-foreground">Enhanced AI-Powered Plant Disease Detection System</p>
          <p className="text-xs text-muted-foreground mt-2">
            Team: Prakash Singh Rawat, Pulkit Pathak, Aman Singh, Mohd Aftab
          </p>
          <p className="text-xs text-muted-foreground mt-1">Mentor: Prof. M Gangadharappa</p>
        </div>
      </footer>
    </div>
  )
}

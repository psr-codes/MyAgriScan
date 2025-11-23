"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface UploadZoneProps {
  onImageSelect: (file: File) => void
  isAnalyzing: boolean
}

export function UploadZone({ onImageSelect, isAnalyzing }: UploadZoneProps) {
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onImageSelect(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault()
    if (e.target.files && e.target.files[0]) {
      onImageSelect(e.target.files[0])
    }
  }

  return (
    <Card
      className={cn(
        "relative flex flex-col items-center justify-center w-full max-w-md p-8 mx-auto text-center border-2 border-dashed transition-all duration-300 cursor-pointer hover:bg-secondary/20",
        dragActive ? "border-primary bg-primary/5" : "border-border",
        isAnalyzing ? "pointer-events-none opacity-50" : "",
      )}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleChange}
        disabled={isAnalyzing}
      />

      <div className="bg-secondary p-4 rounded-full mb-4 group-hover:scale-110 transition-transform">
        {isAnalyzing ? (
          <Loader2 className="w-8 h-8 text-primary animate-spin" />
        ) : (
          <Upload className="w-8 h-8 text-primary" />
        )}
      </div>

      <h3 className="text-lg font-semibold mb-2">{isAnalyzing ? "Analyzing Plant..." : "Upload Plant Photo"}</h3>
      <p className="text-sm text-muted-foreground mb-6">
        Take a photo or drag & drop an image of the affected leaf here.
      </p>

      <Button disabled={isAnalyzing} className="cursor-pointer w-full sm:w-auto">
        {isAnalyzing ? "Processing..." : "Select Image"}
      </Button>
    </Card>
  )
}

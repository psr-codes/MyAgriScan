"use client"

import type { ScanResult } from "@/lib/types"
import { Card } from "@/components/ui/card"
import { Clock } from "lucide-react"

interface HistoryListProps {
  history: ScanResult[]
  onSelect: (scan: ScanResult) => void
}

export function HistoryList({ history, onSelect }: HistoryListProps) {
  if (history.length === 0) return null

  return (
    <div className="w-full max-w-2xl mx-auto mt-12 mb-8">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="w-5 h-5 text-muted-foreground" />
        <h3 className="font-semibold text-muted-foreground">Recent Scans</h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {history.map((scan) => (
          <Card
            key={scan.id}
            className="flex items-center p-3 gap-4 cursor-pointer hover:bg-secondary/50 transition-colors"
            onClick={() => onSelect(scan)}
          >
            <div className="w-12 h-12 rounded-md overflow-hidden bg-muted flex-shrink-0">
              <img
                src={scan.imageUrl || "/placeholder.svg"}
                alt={scan.disease.name}
                className="w-full h-full object-cover"
              />
            </div>
            <div className="min-w-0">
              <p className="font-medium truncate">{scan.disease.name}</p>
              <p className="text-xs text-muted-foreground">{new Date(scan.date).toLocaleDateString()}</p>
            </div>
            <div
              className={`ml-auto w-2 h-2 rounded-full ${scan.disease.name.toLowerCase().includes("healthy") ? "bg-green-500" : "bg-orange-500"}`}
            />
          </Card>
        ))}
      </div>
    </div>
  )
}

export interface DiseaseInfo {
  name: string
  confidence: number
  description: string
  prevention: string[]
  treatment: string[]
  tips: string[]
}

export interface ScanResult {
  id: string
  date: string
  imageUrl: string
  disease: DiseaseInfo
}

import type React from "react"
import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import { Header } from "@/components/plant-guard/header"
const geist = Geist({ subsets: ["latin"] })
const geistMono = Geist_Mono({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "PlantGuard AI - Smart Disease Detection",
  description: "AI-powered plant disease detection and treatment advice for farmers.",
  icons: {
    icon: [{ url: "/favicon.ico", sizes: "any" }],
  },
    generator: 'v0.app'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${geist.className} antialiased min-h-screen flex flex-col`}>
              <Header />
        
        {children}
        {/* Analytics component removed */}
      </body>
    </html>
  )
}

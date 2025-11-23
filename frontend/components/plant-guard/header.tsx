"use client"

import { Leaf, Menu } from "lucide-react"
import { Button } from "@/components/ui/button"
import Link from "next/link"

export function Header() {
  return (
    <header className="sticky top-0 z-50 w-full border-b bg-primary text-primary-foreground shadow-md">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="bg-white/20 p-2 rounded-full">
            <Leaf className="w-6 h-6 md:w-8 md:h-8" />
          </div>
          <div>
            <h1 className="text-xl md:text-2xl font-bold tracking-tight">PlantGuard AI</h1>
            <p className="text-[10px] md:text-xs font-medium text-primary-foreground/80 hidden sm:block">
              Smart Disease Detection for Farmers
            </p>
          </div>
        </div>

        <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
          <Link href="/" className="cursor-pointer hover:text-primary-foreground/80 transition-colors">
            Home
          </Link>
          <button
            onClick={() => window.location.reload()}
            className="cursor-pointer  hover:text-primary-foreground/80 transition-colors"
          >
            New Diagnosis
          </button>
          <Link href="/about" className="cursor-pointer  hover:text-primary-foreground/80 transition-colors">
            About / Help
          </Link>
        </nav>

        <Button variant="ghost" size="icon" className="md:hidden text-primary-foreground">
          <Menu className="w-6 h-6" />
          <span className="sr-only">Toggle menu</span>
        </Button>
      </div>
    </header>
  )
}

"use client"

import type React from "react"

import { useState, useCallback, useEffect } from "react"
import { GripVertical } from "lucide-react"

interface ResizableDividerProps {
  onResize: (width: number) => void
  initialWidth: number
  minWidth?: number
  maxWidth?: number
}

export function ResizableDivider({ onResize, initialWidth, minWidth = 200, maxWidth = 600 }: ResizableDividerProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [startX, setStartX] = useState(0)
  const [startWidth, setStartWidth] = useState(initialWidth)

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      setIsDragging(true)
      setStartX(e.clientX)
      setStartWidth(initialWidth)
      e.preventDefault()
    },
    [initialWidth],
  )

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging) return

      const deltaX = e.clientX - startX
      const newWidth = Math.max(minWidth, Math.min(maxWidth, startWidth + deltaX))
      onResize(newWidth)
    },
    [isDragging, startX, startWidth, minWidth, maxWidth, onResize],
  )

  const handleMouseUp = useCallback(() => {
    setIsDragging(false)
  }, [])

  useEffect(() => {
    if (isDragging) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
      document.body.style.cursor = "col-resize"
      document.body.style.userSelect = "none"

      return () => {
        document.removeEventListener("mousemove", handleMouseMove)
        document.removeEventListener("mouseup", handleMouseUp)
        document.body.style.cursor = ""
        document.body.style.userSelect = ""
      }
    }
  }, [isDragging, handleMouseMove, handleMouseUp])

  return (
    <div
      className={`w-1 bg-gray-200 hover:bg-gray-300 cursor-col-resize flex items-center justify-center group relative min-h-screen ${
        isDragging ? "bg-blue-400" : ""
      }`}
      onMouseDown={handleMouseDown}
    >
      <div className="absolute inset-y-0 -left-1 -right-1 flex items-center justify-center">
        <GripVertical
          className={`h-4 w-4 text-gray-400 group-hover:text-gray-600 transition-colors ${
            isDragging ? "text-blue-600" : ""
          }`}
        />
      </div>
    </div>
  )
}

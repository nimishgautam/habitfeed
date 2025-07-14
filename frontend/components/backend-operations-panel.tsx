"use client"

import type React from "react"

import { useState } from "react"
import { X, Play, Settings, RefreshCw, Download, Brain, Calculator } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { ProgressWindow } from "./progress-window"

interface BackendOperation {
  id: string
  name: string
  description: string
  icon: React.ComponentType<{ className?: string }>
  endpoint: string
  streamEndpoint: string
  color: string
}

interface BackendOperationsPanelProps {
  isOpen: boolean
  onClose: () => void
  onOperationComplete?: () => void
}

// Define available backend operations - easily extensible
const BACKEND_OPERATIONS: BackendOperation[] = [
  {
    id: "update-feeds",
    name: "Update Feeds",
    description: "Update all RSS feeds by fetching latest feed metadata",
    icon: RefreshCw,
    endpoint: "/feeds/update",
    streamEndpoint: "/feeds/update/stream",
    color: "blue",
  },
  {
    id: "fetch-articles",
    name: "Fetch Articles",
    description: "Fetch article content and create embeddings for all articles",
    icon: Download,
    endpoint: "/articles/fetch",
    streamEndpoint: "/articles/fetch/stream",
    color: "green",
  },
  {
    id: "detect-topics",
    name: "Detect Topics",
    description: "Analyze articles and detect topics using machine learning",
    icon: Brain,
    endpoint: "/topics/detect",
    streamEndpoint: "/topics/detect/stream",
    color: "purple",
  },
  {
    id: "recalculate-linucb",
    name: "Recalculate LinUCB",
    description: "Recalculate LinUCB recommendation scores for all articles",
    icon: Calculator,
    endpoint: "/recommendations/linucb/recalculate",
    streamEndpoint: "/recommendations/linucb/recalculate/stream",
    color: "orange",
  },
]

export function BackendOperationsPanel({ isOpen, onClose, onOperationComplete }: BackendOperationsPanelProps) {
  const [useStreaming, setUseStreaming] = useState(true)
  const [runningOperation, setRunningOperation] = useState<string | null>(null)
  const [progressWindow, setProgressWindow] = useState<{
    isOpen: boolean
    title: string
    streamUrl: string
    operationType: "update" | "fetch"
  } | null>(null)

  const handleOperationRun = async (operation: BackendOperation) => {
    if (runningOperation) return // Prevent multiple operations

    setRunningOperation(operation.id)

    try {
      if (useStreaming) {
        // Use streaming version
        setProgressWindow({
          isOpen: true,
          title: operation.name,
          streamUrl: `/api${operation.streamEndpoint}`,
          operationType: operation.id.includes("fetch") ? "fetch" : "update",
        })
      } else {
        // Use non-streaming version
        const response = await fetch(`/api${operation.endpoint}`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({}),
        })

        if (!response.ok) {
          throw new Error(`Failed to ${operation.name.toLowerCase()}: ${response.statusText}`)
        }

        // Show success message
        alert(`${operation.name} completed successfully!`)

        // Call completion callback
        if (onOperationComplete) {
          onOperationComplete()
        }
      }
    } catch (error) {
      console.error(`Failed to run ${operation.name}:`, error)
      alert(`Failed to ${operation.name.toLowerCase()}: ${error instanceof Error ? error.message : "Unknown error"}`)
    } finally {
      if (!useStreaming) {
        setRunningOperation(null)
      }
    }
  }

  const handleProgressWindowClose = () => {
    setProgressWindow(null)
    setRunningOperation(null)

    // Call completion callback
    if (onOperationComplete) {
      onOperationComplete()
    }
  }

  const getColorClasses = (color: string) => {
    const colorMap = {
      blue: "text-blue-600 hover:text-blue-700 hover:bg-blue-50",
      green: "text-green-600 hover:text-green-700 hover:bg-green-50",
      purple: "text-purple-600 hover:text-purple-700 hover:bg-purple-50",
      orange: "text-orange-600 hover:text-orange-700 hover:bg-orange-50",
    }
    return colorMap[color as keyof typeof colorMap] || colorMap.blue
  }

  if (!isOpen) return null

  return (
    <>
      <div
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        onWheel={(e) => e.stopPropagation()}
        onTouchMove={(e) => e.stopPropagation()}
      >
        <div
          className="bg-white rounded-lg shadow-xl w-full max-w-2xl mx-4 max-h-[80vh] overflow-hidden flex flex-col"
          onWheel={(e) => e.stopPropagation()}
          onTouchMove={(e) => e.stopPropagation()}
        >
          <div className="flex items-center justify-between p-6 border-b border-gray-200 flex-shrink-0">
            <div className="flex items-center gap-2">
              <Settings className="h-5 w-5 text-gray-600" />
              <h2 className="text-lg font-semibold text-gray-900">Backend Operations</h2>
            </div>
            <Button variant="ghost" size="icon" onClick={onClose} disabled={!!runningOperation}>
              <X className="h-4 w-4" />
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto">
            <div className="p-6 space-y-6">
              {/* Streaming Toggle */}
              <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                <div className="space-y-1">
                  <Label htmlFor="streaming-toggle" className="text-sm font-medium">
                    Use Streaming
                  </Label>
                  <p className="text-xs text-gray-600">Show real-time progress updates during operations</p>
                </div>
                <Switch
                  id="streaming-toggle"
                  checked={useStreaming}
                  onCheckedChange={setUseStreaming}
                  disabled={!!runningOperation}
                />
              </div>

              {/* Operations List */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-gray-700 uppercase tracking-wide">Available Operations</h3>

                <div className="grid gap-3">
                  {BACKEND_OPERATIONS.map((operation) => {
                    const Icon = operation.icon
                    const isRunning = runningOperation === operation.id
                    const isDisabled = !!runningOperation && !isRunning

                    return (
                      <div
                        key={operation.id}
                        className={`flex items-center justify-between p-4 border rounded-lg transition-colors ${
                          isDisabled ? "bg-gray-50 opacity-50" : "bg-white hover:bg-gray-50"
                        }`}
                      >
                        <div className="flex items-start gap-3 flex-1">
                          <Icon className={`h-5 w-5 mt-0.5 ${getColorClasses(operation.color).split(" ")[0]}`} />
                          <div className="space-y-1">
                            <h4 className="text-sm font-medium text-gray-900">{operation.name}</h4>
                            <p className="text-xs text-gray-600">{operation.description}</p>
                          </div>
                        </div>

                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleOperationRun(operation)}
                          disabled={isDisabled}
                          className={`gap-2 ${getColorClasses(operation.color)}`}
                        >
                          {isRunning ? (
                            <>
                              <div className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                              Running...
                            </>
                          ) : (
                            <>
                              <Play className="h-4 w-4" />
                              Run
                            </>
                          )}
                        </Button>
                      </div>
                    )
                  })}
                </div>
              </div>

              {/* Info Section */}
              <div className="p-4 bg-blue-50 rounded-lg">
                <div className="flex gap-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0" />
                  <div className="text-sm text-blue-800">
                    <p className="font-medium mb-1">Operation Notes:</p>
                    <ul className="space-y-1 text-xs">
                      <li>• Operations run sequentially - only one can run at a time</li>
                      <li>• Streaming mode shows real-time progress and logs</li>
                      <li>• Non-streaming mode runs in background with simple confirmation</li>
                      <li>• Some operations may take several minutes to complete</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Progress Window for Streaming Operations */}
      {progressWindow && (
        <ProgressWindow
          isOpen={progressWindow.isOpen}
          onClose={handleProgressWindowClose}
          title={progressWindow.title}
          streamUrl={progressWindow.streamUrl}
          operationType={progressWindow.operationType}
        />
      )}
    </>
  )
}

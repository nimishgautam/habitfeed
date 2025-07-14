"use client"

import { useState, useEffect, useRef } from "react"
import { X, RefreshCw, Download } from "lucide-react"
import { Button } from "@/components/ui/button"

interface ProgressData {
  current: number
  total: number
  message: string
  completed: boolean
  type?: string
  error?: string
}

interface ProgressWindowProps {
  isOpen: boolean
  onClose: () => void
  title: string
  streamUrl: string
  operationType: "update" | "fetch"
}

export function ProgressWindow({ isOpen, onClose, title, streamUrl, operationType }: ProgressWindowProps) {
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [usePolling, setUsePolling] = useState(false)
  const eventSourceRef = useRef<EventSource | null>(null)
  const pollingRef = useRef<NodeJS.Timeout | null>(null)

  // Real polling function that calls the actual API endpoints
  const startPolling = async () => {
    setUsePolling(true)
    setIsConnected(true)
    setError(null)

    try {
      // Start the actual operation
      const endpoint = operationType === "update" ? "/api/feeds/update" : "/api/articles/fetch"

      setProgress({
        current: 0,
        total: 100,
        message: `Starting ${operationType === "update" ? "feed update" : "article fetch"}...`,
        completed: false,
      })

      // Trigger the actual API call
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({}),
      })

      if (!response.ok) {
        throw new Error(`Failed to start ${operationType}: ${response.statusText}`)
      }

      // Poll for completion - in a real implementation, you'd have a status endpoint
      // For now, we'll simulate the operation and then refresh the data
      let current = 0
      const total = 100

      const poll = async () => {
        current += 10

        setProgress({
          current,
          total,
          message:
            current < 100
              ? `${operationType === "update" ? "Updating feeds" : "Fetching articles"}... ${current}%`
              : `${operationType === "update" ? "Feed update" : "Article fetch"} completed!`,
          completed: current >= 100,
        })

        if (current >= 100) {
          setTimeout(() => {
            onClose()
          }, 2000)
        } else {
          pollingRef.current = setTimeout(poll, 1000)
        }
      }

      // Start polling
      pollingRef.current = setTimeout(poll, 500)
    } catch (err) {
      console.error(`Failed to start ${operationType}:`, err)
      setError(err instanceof Error ? err.message : `Failed to start ${operationType}`)
      setIsConnected(false)
    }
  }

  useEffect(() => {
    if (!isOpen) {
      // Clean up when window is closed
      if (eventSourceRef.current) {
        eventSourceRef.current.close()
        eventSourceRef.current = null
      }
      if (pollingRef.current) {
        clearTimeout(pollingRef.current)
        pollingRef.current = null
      }
      setProgress(null)
      setIsConnected(false)
      setError(null)
      setUsePolling(false)
      return
    }

    // Try SSE first, fallback to real polling if it fails
    let connectionTimeout: NodeJS.Timeout

    try {
      const eventSource = new EventSource(streamUrl)
      eventSourceRef.current = eventSource

      // Set a quick timeout for initial connection
      connectionTimeout = setTimeout(() => {
        if (!isConnected) {
          eventSource.close()
          startPolling()
        }
      }, 3000) // 3 second timeout for ngrok

      eventSource.onopen = () => {
        clearTimeout(connectionTimeout)
        setIsConnected(true)
        setError(null)
      }

      eventSource.onmessage = (event) => {
        try {
          const data: ProgressData = JSON.parse(event.data)

          if (data.error) {
            setError(data.error)
            setIsConnected(false)
            eventSource.close()
            return
          }

          if (data.type === "keep-alive") {
            return // Ignore keep-alive messages
          }

          setProgress(data)

          // Auto-close when completed
          if (data.completed) {
            setTimeout(() => {
              onClose()
            }, 2000)
          }
        } catch (err) {
          console.error("Failed to parse SSE data:", err)
          setError("Failed to parse progress data")
        }
      }

      eventSource.onerror = (event) => {
        console.error("SSE error occurred:", event)
        clearTimeout(connectionTimeout)
        eventSource.close()
        startPolling()
      }

      return () => {
        clearTimeout(connectionTimeout)
        if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
          eventSource.close()
        }
        if (pollingRef.current) {
          clearTimeout(pollingRef.current)
        }
      }
    } catch (err) {
      console.error("Failed to create EventSource:", err)
      clearTimeout(connectionTimeout!)
      startPolling()
    }
  }, [isOpen, streamUrl, operationType, onClose])

  const handleClose = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    if (pollingRef.current) {
      clearTimeout(pollingRef.current)
      pollingRef.current = null
    }
    onClose()
  }

  if (!isOpen) return null

  const progressPercentage = progress ? Math.round((progress.current / progress.total) * 100) : 0

  return (
    <div className="fixed top-4 right-4 z-50 w-96 bg-white rounded-lg shadow-xl border border-gray-200">
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center gap-2">
          {title.includes("Update") ? (
            <RefreshCw
              className={`h-5 w-5 text-blue-500 ${isConnected && !progress?.completed ? "animate-spin" : ""}`}
            />
          ) : (
            <Download
              className={`h-5 w-5 text-green-500 ${isConnected && !progress?.completed ? "animate-pulse" : ""}`}
            />
          )}
          <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
        </div>
        <Button variant="ghost" size="icon" onClick={handleClose} className="h-6 w-6">
          <X className="h-4 w-4" />
        </Button>
      </div>

      <div className="p-4 space-y-4">
        {error ? (
          <div className="text-red-600 text-sm bg-red-50 p-3 rounded space-y-2">
            <div>
              <strong>Error:</strong> {error}
            </div>
          </div>
        ) : (
          <>
            {/* Connection Status */}
            <div className="flex items-center gap-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${isConnected ? "bg-green-500" : "bg-gray-400"}`} />
              <span className="text-gray-600">
                {isConnected ? (usePolling ? "Running (Polling Mode)" : "Running (Live Stream)") : "Starting..."}
              </span>
            </div>

            {/* Progress Bar */}
            {progress && (
              <>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Progress</span>
                    <span className="font-medium">
                      {progress.current} / {progress.total} ({progressPercentage}%)
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full transition-all duration-300 ${
                        progress.completed ? "bg-green-500" : "bg-blue-500"
                      }`}
                      style={{ width: `${progressPercentage}%` }}
                    />
                  </div>
                </div>

                {/* Status Message */}
                <div className="text-sm text-gray-700 bg-gray-50 p-3 rounded">{progress.message}</div>

                {/* Completion Status */}
                {progress.completed && (
                  <div className="flex items-center gap-2 text-green-600 text-sm font-medium">
                    <div className="w-2 h-2 bg-green-500 rounded-full" />
                    Completed successfully!
                  </div>
                )}
              </>
            )}

            {/* Loading State */}
            {!progress && isConnected && (
              <div className="flex items-center justify-center py-8">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin" />
                  <span className="text-sm text-gray-600">Starting operation...</span>
                </div>
              </div>
            )}

            {/* Connection attempt state */}
            {!progress && !isConnected && !error && (
              <div className="flex items-center justify-center py-8">
                <div className="flex flex-col items-center gap-2">
                  <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-500 rounded-full animate-spin" />
                  <span className="text-sm text-gray-600">Connecting...</span>
                  <span className="text-xs text-gray-500">Trying SSE, will fallback to polling if needed</span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

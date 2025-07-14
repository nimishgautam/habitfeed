"use client"

import type React from "react"

import { useState } from "react"
import { X, Plus, Rss } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { addFeed } from "../lib/api"
import type { FeedInfoSchema } from "../lib/api"

interface AddFeedModalProps {
  isOpen: boolean
  onClose: () => void
  onAddFeed: (feed: FeedInfoSchema) => void
}

export function AddFeedModal({ isOpen, onClose, onAddFeed }: AddFeedModalProps) {
  const [feedUrl, setFeedUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!feedUrl.trim()) return

    setIsLoading(true)
    setError(null)

    try {
      const newFeed = await addFeed(feedUrl.trim())
      onAddFeed(newFeed)
      setFeedUrl("")
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add feed")
    } finally {
      setIsLoading(false)
    }
  }

  const handleClose = () => {
    if (!isLoading) {
      setFeedUrl("")
      setError(null)
      onClose()
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md mx-4">
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center gap-2">
            <Rss className="h-5 w-5 text-orange-500" />
            <h2 className="text-lg font-semibold text-gray-900">Add RSS Feed</h2>
          </div>
          <Button variant="ghost" size="icon" onClick={handleClose} disabled={isLoading}>
            <X className="h-4 w-4" />
          </Button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-4">
          <div>
            <Label htmlFor="feedUrl" className="text-sm font-medium text-gray-700">
              RSS URL *
            </Label>
            <Input
              id="feedUrl"
              type="url"
              value={feedUrl}
              onChange={(e) => setFeedUrl(e.target.value)}
              placeholder="https://example.com/rss.xml"
              className="mt-1"
              disabled={isLoading}
              required
            />
          </div>

          {error && <div className="text-sm text-red-600 bg-red-50 p-2 rounded">{error}</div>}

          <div className="flex gap-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
              disabled={isLoading}
              className="flex-1 bg-transparent"
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isLoading || !feedUrl.trim()} className="flex-1">
              {isLoading ? (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Adding...
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <Plus className="h-4 w-4" />
                  Add Feed
                </div>
              )}
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}

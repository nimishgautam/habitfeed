"use client"

import Image from "next/image"
import { ArrowLeft, ExternalLink, ThumbsUp, Bookmark, Meh, SkipForward, ThumbsDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { formatRelativeTime, recordArticleAction, getArticleActions, type ArticleSchema } from "../lib/api"
import { useState, useEffect } from "react"

// Define extended article type locally
export interface ExtendedArticle extends ArticleSchema {
  source?: string
  author?: string
  thumbnail?: string
}

interface ArticleDetailViewProps {
  article: ExtendedArticle
  onBack: (shouldRefresh?: boolean) => void
  isLoading?: boolean
}

export function ArticleDetailView({ article, onBack, isLoading = false }: ArticleDetailViewProps) {
  const [actionFeedback, setActionFeedback] = useState<string | null>(null)
  const [existingActions, setExistingActions] = useState<Set<string>>(new Set())
  const [isLoadingActions, setIsLoadingActions] = useState(false)

  // Load existing actions when article changes
  useEffect(() => {
    const loadActions = async () => {
      if (!article?.article_id) return

      setIsLoadingActions(true)
      try {
        const actions = await getArticleActions(article.article_id)
        const actionTypes = new Set(actions.map((action) => action.action))
        setExistingActions(actionTypes)
      } catch (error) {
        console.error("Failed to load article actions:", error)
      } finally {
        setIsLoadingActions(false)
      }
    }

    loadActions()
  }, [article?.article_id])

  const handleAction = async (action: string) => {
    try {
      const response = await recordArticleAction(article.article_id, action)
      setActionFeedback(`${action} recorded successfully`)

      // Update existing actions
      setExistingActions((prev) => new Set([...prev, action]))

      // For negative actions, return to feed after showing feedback and refresh
      if (action === "meh" || action === "skip" || action === "downvote") {
        setTimeout(() => {
          setActionFeedback(null)
          onBack(true) // Pass true to indicate refresh is needed
        }, 1500) // Show feedback for 1.5 seconds then navigate back
      } else {
        // For positive actions, just clear feedback after 2 seconds
        setTimeout(() => setActionFeedback(null), 2000)
      }
    } catch (error) {
      console.error(`Failed to record ${action}:`, error)
      setActionFeedback(`Failed to record ${action}`)
      setTimeout(() => setActionFeedback(null), 2000)
    }
  }

  if (isLoading) {
    return (
      <div className="flex-1 bg-white flex items-center justify-center">
        <div className="flex flex-col items-center">
          <div className="w-10 h-10 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
          <p className="text-gray-600">Loading article...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 bg-white">
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <Button variant="ghost" size="sm" onClick={onBack} className="gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to feed
          </Button>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => window.open(article.url, "_blank")}
              title="Open original article"
            >
              <ExternalLink className="h-4 w-4" />
            </Button>
            <div className="w-px h-6 bg-gray-300 mx-1" />
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleAction("upvote")}
              title="Upvote"
              className={`${
                existingActions.has("upvote")
                  ? "text-green-700 bg-green-100 hover:bg-green-200"
                  : "text-green-600 hover:text-green-700 hover:bg-green-50"
              }`}
            >
              <ThumbsUp className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleAction("save")}
              title="Save article"
              className={`${
                existingActions.has("save")
                  ? "text-blue-700 bg-blue-100 hover:bg-blue-200"
                  : "text-blue-600 hover:text-blue-700 hover:bg-blue-50"
              }`}
            >
              <Bookmark className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleAction("meh")}
              title="Meh"
              className={`${
                existingActions.has("meh")
                  ? "text-yellow-700 bg-yellow-100 hover:bg-yellow-200"
                  : "text-yellow-600 hover:text-yellow-700 hover:bg-yellow-50"
              }`}
            >
              <Meh className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleAction("skip")}
              title="Skip"
              className={`${
                existingActions.has("skip")
                  ? "text-orange-700 bg-orange-100 hover:bg-orange-200"
                  : "text-orange-600 hover:text-orange-700 hover:bg-orange-50"
              }`}
            >
              <SkipForward className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => handleAction("downvote")}
              title="Downvote"
              className={`${
                existingActions.has("downvote")
                  ? "text-red-700 bg-red-100 hover:bg-red-200"
                  : "text-red-600 hover:text-red-700 hover:bg-red-50"
              }`}
            >
              <ThumbsDown className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
      {actionFeedback && (
        <div className="px-4 py-2 bg-blue-50 border-b border-blue-200">
          <p className="text-sm text-blue-700">{actionFeedback}</p>
        </div>
      )}
      <div className="overflow-y-auto">
        <article className="max-w-4xl mx-auto p-8">
          <header className="mb-8">
            <h1 className="text-3xl font-bold text-gray-900 mb-4">{article.title}</h1>
            <div className="flex items-center gap-4 text-sm text-gray-600 mb-4">
              {article.author && <span className="font-medium">{article.author}</span>}
              {article.author && <span>•</span>}
              <span>{article.source || new URL(article.url).hostname.replace("www.", "")}</span>
              <span>•</span>
              <span>{article.pub_date ? formatRelativeTime(article.pub_date) : "Unknown"}</span>
            </div>

            {article.thumbnail && (
              <div className="mb-6">
                <Image
                  src={article.thumbnail || "/placeholder.svg"}
                  alt=""
                  width={800}
                  height={400}
                  className="rounded-lg object-cover w-full"
                />
              </div>
            )}
          </header>

          <div className="prose prose-lg max-w-none">
            <p className="text-gray-700 leading-relaxed whitespace-pre-line">
              {article.full_text || article.rss_description}
            </p>

            <div className="mt-8 pt-4 border-t border-gray-200">
              <a
                href={article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 text-blue-600 hover:text-blue-800"
              >
                Read full article <ExternalLink className="h-4 w-4" />
              </a>
            </div>
          </div>
        </article>
      </div>
    </div>
  )
}

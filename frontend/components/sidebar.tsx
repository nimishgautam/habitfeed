"use client"

import { useState } from "react"
import {
  ChevronRight,
  ChevronDown,
  MoreHorizontal,
  RefreshCw,
  Settings,
  Plus,
  Download,
  Heart,
  Bookmark,
  Award,
  GitCompare,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { AddFeedModal } from "./add-feed-modal"
import { BackendOperationsPanel } from "./backend-operations-panel"
import type { FeedInfoSchema, ArticleSchema } from "../lib/api"

// Define extended types locally since we removed mock-data
export interface ExtendedArticle extends ArticleSchema {
  source?: string
  author?: string
  thumbnail?: string
}

export interface ExtendedFeed extends FeedInfoSchema {
  name: string
  count: number
  articles: ExtendedArticle[]
}

interface SidebarProps {
  feeds: ExtendedFeed[]
  selectedFeed: number | null
  selectedArticle: number | null
  selectedActionFeed: string | null
  onFeedSelect: (feedId: number) => void
  onArticleSelect: (articleId: number) => void
  onActionFeedSelect: (action: string | null) => void
  onAddFeed: (feed: FeedInfoSchema) => void
  onRefreshFeeds: () => Promise<void>
  onFeedExpand?: (feedId: number) => void
  width: number
  isLoading?: boolean
}

export function Sidebar({
  feeds,
  selectedFeed,
  selectedArticle,
  selectedActionFeed,
  onFeedSelect,
  onArticleSelect,
  onActionFeedSelect,
  onAddFeed,
  onRefreshFeeds,
  onFeedExpand,
  width,
  isLoading = false,
}: SidebarProps) {
  const [expandedFeeds, setExpandedFeeds] = useState<Set<number>>(new Set())
  const [isAddFeedModalOpen, setIsAddFeedModalOpen] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isBackendPanelOpen, setIsBackendPanelOpen] = useState(false)

  const toggleFeed = (feedId: number) => {
    const newExpanded = new Set(expandedFeeds)
    if (newExpanded.has(feedId)) {
      newExpanded.delete(feedId)
    } else {
      newExpanded.add(feedId)
      // Load articles when expanding if callback is provided
      if (onFeedExpand) {
        onFeedExpand(feedId)
      }
    }
    setExpandedFeeds(newExpanded)
  }

  const handleFeedClick = (feedId: number) => {
    onFeedSelect(feedId)
    onActionFeedSelect(null) // Clear action feed selection
    if (!expandedFeeds.has(feedId)) {
      toggleFeed(feedId)
    }
  }

  const handleActionFeedClick = (action: string) => {
    if (selectedActionFeed === action) {
      onActionFeedSelect(null) // Deselect if already selected
    } else {
      // Only call onActionFeedSelect - don't call onFeedSelect!
      onActionFeedSelect(action)
      // The main component will handle clearing other selections
    }
  }

  const handleRefresh = async () => {
    setIsRefreshing(true)
    try {
      await onRefreshFeeds()
    } catch (error) {
      console.error("Failed to refresh feeds:", error)
    } finally {
      setIsRefreshing(false)
    }
  }

  const handleOpenBackendPanel = () => {
    setIsBackendPanelOpen(true)
  }

  return (
    <div className="bg-gray-50 border-r border-gray-200 flex flex-col" style={{ width: `${width}px` }}>
      <div className="p-4 border-b border-gray-200 flex-shrink-0">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Feeds</h2>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" className="h-6 w-6">
              <Settings className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={handleRefresh}
              disabled={isRefreshing || isLoading}
              title="Refresh Feeds"
            >
              <RefreshCw className={`h-4 w-4 ${isRefreshing ? "animate-spin" : ""}`} />
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={handleOpenBackendPanel}
              disabled={isLoading}
              title="Backend Operations"
            >
              <Download className="h-4 w-4" />
            </Button>
            <Button variant="ghost" size="icon" className="h-6 w-6">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="p-2 border-b border-gray-200 flex-shrink-0">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsAddFeedModalOpen(true)}
          className="w-full justify-start gap-2 text-gray-700 hover:text-gray-900"
          disabled={isLoading}
        >
          <Plus className="h-4 w-4" />
          Add Feed
        </Button>
      </div>

      {/* Scrollable content area */}
      <div className="flex-1">
        {/* Action Feeds Section */}
        <div className="border-b border-gray-200">
          <div className="p-2 space-y-1">
            <button
              className={`w-full flex items-center gap-2 px-2 py-2 text-sm rounded hover:bg-gray-200 text-left ${
                selectedActionFeed === "recommended" ? "bg-blue-100 text-blue-700" : "text-gray-700"
              }`}
              onClick={() => handleActionFeedClick("recommended")}
            >
              <Award className="h-4 w-4" />
              <span>Recommended Articles</span>
            </button>

            <button
              className={`w-full flex items-center gap-2 px-2 py-2 text-sm rounded hover:bg-gray-200 text-left ${
                selectedActionFeed === "similar" ? "bg-blue-100 text-blue-700" : "text-gray-700"
              }`}
              onClick={() => handleActionFeedClick("similar")}
            >
              <GitCompare className="h-4 w-4" />
              <span>Similar Articles</span>
            </button>

            <button
              className={`w-full flex items-center gap-2 px-2 py-2 text-sm rounded hover:bg-gray-200 text-left ${
                selectedActionFeed === "recent" ? "bg-blue-100 text-blue-700" : "text-gray-700"
              }`}
              onClick={() => handleActionFeedClick("recent")}
            >
              <RefreshCw className="h-4 w-4" />
              <span>Recent Articles</span>
            </button>

            <button
              className={`w-full flex items-center gap-2 px-2 py-2 text-sm rounded hover:bg-gray-200 text-left ${
                selectedActionFeed === "save" ? "bg-blue-100 text-blue-700" : "text-gray-700"
              }`}
              onClick={() => handleActionFeedClick("save")}
            >
              <Bookmark className="h-4 w-4" />
              <span>Saved Articles</span>
            </button>

            <button
              className={`w-full flex items-center gap-2 px-2 py-2 text-sm rounded hover:bg-gray-200 text-left ${
                selectedActionFeed === "upvote" ? "bg-blue-100 text-blue-700" : "text-gray-700"
              }`}
              onClick={() => handleActionFeedClick("upvote")}
            >
              <Heart className="h-4 w-4" />
              <span>Upvoted Articles</span>
            </button>
          </div>
        </div>

        {/* Feeds Section */}
        <div className="p-2">
          {isLoading ? (
            <div className="flex justify-center py-4">
              <div className="w-6 h-6 border-2 border-gray-300 border-t-blue-600 rounded-full animate-spin"></div>
            </div>
          ) : (
            feeds.map((feed) => (
              <div key={feed.feed_id} className="mb-1">
                <div className="flex items-center group">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6 p-0 mr-1"
                    onClick={() => toggleFeed(feed.feed_id)}
                  >
                    {expandedFeeds.has(feed.feed_id) ? (
                      <ChevronDown className="h-3 w-3" />
                    ) : (
                      <ChevronRight className="h-3 w-3" />
                    )}
                  </Button>
                  <button
                    className={`flex-1 flex items-center justify-between px-2 py-1 text-sm rounded hover:bg-gray-200 text-left ${
                      selectedFeed === feed.feed_id ? "bg-blue-100 text-blue-700" : "text-gray-700"
                    }`}
                    onClick={() => handleFeedClick(feed.feed_id)}
                  >
                    <span>{feed.title || feed.name}</span>
                    {feed.articles.length > 0 && <span className="text-xs text-gray-500">{feed.count}</span>}
                  </button>
                </div>

                {expandedFeeds.has(feed.feed_id) && (
                  <div className="ml-6 mt-1">
                    {feed.articles.length === 0 ? (
                      <div className="px-2 py-1 text-xs text-gray-500">Loading articles...</div>
                    ) : (
                      feed.articles.map((article) => (
                        <button
                          key={article.article_id}
                          className={`block w-full px-2 py-1 text-sm text-left rounded hover:bg-gray-200 truncate ${
                            selectedArticle === article.article_id ? "bg-blue-100 text-blue-700" : "text-gray-600"
                          }`}
                          onClick={() => onArticleSelect(article.article_id)}
                          title={article.title || ""}
                        >
                          {article.title}
                        </button>
                      ))
                    )}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
      <AddFeedModal isOpen={isAddFeedModalOpen} onClose={() => setIsAddFeedModalOpen(false)} onAddFeed={onAddFeed} />
      <BackendOperationsPanel
        isOpen={isBackendPanelOpen}
        onClose={() => setIsBackendPanelOpen(false)}
        onOperationComplete={onRefreshFeeds}
      />
    </div>
  )
}

"use client"

import type React from "react"

import Image from "next/image"
import { MoreHorizontal, RefreshCw, SkipForward } from "lucide-react"
import { Button } from "@/components/ui/button"
import { formatRelativeTime, recordArticleAction, type ArticleSchema } from "../lib/api"

// Define extended article type locally
export interface ExtendedArticle extends ArticleSchema {
  source?: string
  author?: string
  thumbnail?: string
}

// Helper function to safely render HTML content
function createMarkup(html: string) {
  return { __html: html }
}

// Helper function to strip HTML tags for plain text fallback
function stripHtml(html: string): string {
  return html.replace(/<[^>]*>/g, "")
}

interface ArticleListViewProps {
  articles: ExtendedArticle[]
  feedName: string
  onArticleSelect: (articleId: number) => void
  onArticleSkip?: (articleId: number) => void
  onRefresh?: () => void
  isLoading?: boolean
}

export function ArticleListView({
  articles,
  feedName,
  onArticleSelect,
  onArticleSkip,
  onRefresh,
  isLoading = false,
}: ArticleListViewProps) {
  const handleSkip = async (articleId: number, e: React.MouseEvent) => {
    e.stopPropagation() // Prevent article selection
    try {
      await recordArticleAction(articleId, "skip")
      if (onArticleSkip) {
        onArticleSkip(articleId)
      }
    } catch (error) {
      console.error("Failed to skip article:", error)
    }
  }

  return (
    <div className="flex-1 bg-white">
      <div className="border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl font-semibold text-gray-900">{feedName}</h1>
          <div className="flex items-center gap-2">
            {onRefresh && (
              <Button
                variant="ghost"
                size="icon"
                onClick={onRefresh}
                disabled={isLoading}
                className={isLoading ? "animate-spin" : ""}
              >
                <RefreshCw className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="icon">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      <div className="overflow-y-auto">
        {articles.length === 0 ? (
          <div className="p-8 text-center text-gray-500">
            {isLoading ? (
              <div className="flex flex-col items-center">
                <div className="w-8 h-8 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mb-4"></div>
                <p>Loading articles...</p>
              </div>
            ) : (
              "No articles available in this feed."
            )}
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {articles.map((article) => (
              <article
                key={article.article_id}
                className="p-4 hover:bg-gray-50 cursor-pointer transition-colors"
                onClick={() => onArticleSelect(article.article_id)}
              >
                <div className="flex gap-4">
                  {article.thumbnail && (
                    <div className="flex-shrink-0">
                      <Image
                        src={article.thumbnail || "/placeholder.svg"}
                        alt=""
                        width={80}
                        height={80}
                        className="rounded-lg object-cover"
                      />
                    </div>
                  )}
                  <div className="flex-1 min-w-0">
                    <h2 className="text-lg font-semibold text-gray-900 mb-2 line-clamp-2">{article.title}</h2>
                    <div className="flex items-center gap-2 text-sm text-gray-600 mb-2">
                      {article.author && <span className="font-medium">{article.author}</span>}
                      {article.author && <span>•</span>}
                      <span>{article.source || new URL(article.url).hostname.replace("www.", "")}</span>
                      <span>•</span>
                      <span>{article.pub_date ? formatRelativeTime(article.pub_date) : "Unknown"}</span>
                    </div>
                    <div
                      className="text-gray-700 text-sm line-clamp-3 mb-3 prose prose-sm max-w-none"
                      dangerouslySetInnerHTML={
                        article.rss_description ? createMarkup(article.rss_description) : undefined
                      }
                    />

                    {/* Skip button */}
                    <div className="flex justify-end mb-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => handleSkip(article.article_id, e)}
                        className="text-orange-600 hover:text-orange-700 hover:bg-orange-50 gap-1"
                        title="Skip this article"
                      >
                        <SkipForward className="h-3 w-3" />
                        Skip
                      </Button>
                    </div>
                  </div>
                </div>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

"use client"

import { useState, useEffect } from "react"
import { Sidebar, type ExtendedFeed, type ExtendedArticle } from "./components/sidebar"
import { ArticleListView } from "./components/article-list-view"
import { ArticleDetailView } from "./components/article-detail-view"
import { ResizableDivider } from "./components/resizable-divider"
import {
  getAllFeeds,
  getArticlesByFeed,
  getRecentArticles,
  getRecommendedArticles,
  getSimilarArticles,
  getArticleById,
  getArticlesByAction,
  extractSourceFromUrl,
  type FeedInfoSchema,
  type ArticleSchema,
} from "./lib/api"

export default function HabitFeed() {
  const [feeds, setFeeds] = useState<ExtendedFeed[]>([])
  const [selectedFeed, setSelectedFeed] = useState<number | null>(null)
  const [selectedArticle, setSelectedArticle] = useState<number | null>(null)
  const [selectedActionFeed, setSelectedActionFeed] = useState<string | null>(null)
  const [sidebarWidth, setSidebarWidth] = useState(320)
  const [currentArticles, setCurrentArticles] = useState<ExtendedArticle[]>([])
  const [currentArticleDetail, setCurrentArticleDetail] = useState<ExtendedArticle | null>(null)
  const [isLoadingFeeds, setIsLoadingFeeds] = useState(true)
  const [isLoadingArticles, setIsLoadingArticles] = useState(false)
  const [isLoadingArticleDetail, setIsLoadingArticleDetail] = useState(false)

  // Fetch all feeds on component mount
  useEffect(() => {
    loadFeeds()
  }, [])

  // Load feeds from API
  const loadFeeds = async () => {
    setIsLoadingFeeds(true)
    try {
      const feedsData = await getAllFeeds()

      // Convert API feeds to ExtendedFeed format
      const extendedFeeds: ExtendedFeed[] = feedsData.map((feed) => ({
        ...feed,
        name: feed.title || `Feed ${feed.feed_id}`,
        count: 0,
        articles: [],
      }))

      setFeeds(extendedFeeds)

      // Only load recommended articles if no other selection is active
      if (extendedFeeds.length > 0 && !selectedActionFeed && !selectedFeed) {
        setSelectedActionFeed("recommended") // Set recommended as default
        loadRecommendedArticles()
      }
    } catch (error) {
      console.error("Failed to load feeds:", error)
    } finally {
      setIsLoadingFeeds(false)
    }
  }

  // Load recent articles
  const loadRecentArticles = async () => {
    setIsLoadingArticles(true)
    try {
      const articles = await getRecentArticles(20)

      // Convert API articles to ExtendedArticle format
      const extendedArticles: ExtendedArticle[] = articles.map((article) => ({
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }))

      setCurrentArticles(extendedArticles)
    } catch (error) {
      console.error("Failed to load recent articles:", error)
    } finally {
      setIsLoadingArticles(false)
    }
  }

  // Load recommended articles
  const loadRecommendedArticles = async () => {
    setIsLoadingArticles(true)
    try {
      const articles = await getRecommendedArticles(20)

      // Convert API articles to ExtendedArticle format
      const extendedArticles: ExtendedArticle[] = articles.map((article) => ({
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }))

      setCurrentArticles(extendedArticles)
    } catch (error) {
      console.error("Failed to load recommended articles:", error)
    } finally {
      setIsLoadingArticles(false)
    }
  }

  // Add a new function to load similar articles
  const loadSimilarArticles = async () => {
    setIsLoadingArticles(true)
    try {
      const articles = await getSimilarArticles(20)

      // Convert API articles to ExtendedArticle format
      const extendedArticles: ExtendedArticle[] = articles.map((article) => ({
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }))

      setCurrentArticles(extendedArticles)
    } catch (error) {
      console.error("Failed to load similar articles:", error)
    } finally {
      setIsLoadingArticles(false)
    }
  }

  // Load articles for a specific feed
  const loadFeedArticles = async (feedId: number) => {
    setIsLoadingArticles(true)
    try {
      const articles = await getArticlesByFeed(feedId)

      // Convert API articles to ExtendedArticle format
      const extendedArticles: ExtendedArticle[] = articles.map((article) => ({
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }))

      setCurrentArticles(extendedArticles)

      // Update feed count and articles in the sidebar
      setFeeds((prevFeeds) =>
        prevFeeds.map((feed) =>
          feed.feed_id === feedId ? { ...feed, count: articles.length, articles: extendedArticles } : feed,
        ),
      )
    } catch (error) {
      console.error(`Failed to load articles for feed ${feedId}:`, error)
    } finally {
      setIsLoadingArticles(false)
    }
  }

  // Load articles for sidebar expansion (doesn't change main view)
  const loadFeedArticlesForSidebar = async (feedId: number) => {
    try {
      const articles = await getArticlesByFeed(feedId)

      // Convert API articles to ExtendedArticle format
      const extendedArticles: ExtendedArticle[] = articles.map((article) => ({
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }))

      // Update only the feed's articles array for sidebar display
      setFeeds((prevFeeds) =>
        prevFeeds.map((feed) =>
          feed.feed_id === feedId ? { ...feed, count: articles.length, articles: extendedArticles } : feed,
        ),
      )
    } catch (error) {
      console.error(`Failed to load articles for sidebar expansion ${feedId}:`, error)
    }
  }

  // Load articles by action type
  const loadActionFeedArticles = async (action: string) => {
    setIsLoadingArticles(true)
    try {
      let articles: ArticleSchema[] = []

      if (action === "recent") {
        articles = await getRecentArticles(20)
      } else if (action === "recommended") {
        articles = await getRecommendedArticles(20)
      } else if (action === "similar") {
        articles = await getSimilarArticles(20)
      } else {
        articles = await getArticlesByAction(action)
      }

      // Convert API articles to ExtendedArticle format
      const extendedArticles: ExtendedArticle[] = articles.map((article) => ({
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }))

      setCurrentArticles(extendedArticles)
    } catch (error) {
      console.error(`Failed to load articles for action ${action}:`, error)
    } finally {
      setIsLoadingArticles(false)
    }
  }

  // Load a specific article
  const loadArticleDetail = async (articleId: number) => {
    setIsLoadingArticleDetail(true)
    try {
      const article = await getArticleById(articleId)

      // Convert API article to ExtendedArticle format
      const extendedArticle: ExtendedArticle = {
        ...article,
        source: extractSourceFromUrl(article.url),
        author: article.title?.split(" - ").pop() || undefined,
      }

      setCurrentArticleDetail(extendedArticle)
    } catch (error) {
      console.error(`Failed to load article ${articleId}:`, error)
    } finally {
      setIsLoadingArticleDetail(false)
    }
  }

  // Refresh all feeds - simplified version
  const refreshFeeds = async () => {
    await loadFeeds()
    // Reload current view based on current selection
    if (selectedActionFeed === "recommended") {
      loadRecommendedArticles()
    } else if (selectedActionFeed === "similar") {
      loadSimilarArticles()
    } else if (selectedActionFeed === "recent") {
      loadRecentArticles()
    } else if (selectedActionFeed) {
      loadActionFeedArticles(selectedActionFeed)
    } else if (selectedFeed && selectedFeed > 0) {
      loadFeedArticles(selectedFeed)
    } else {
      setSelectedActionFeed("recommended")
      loadRecommendedArticles()
    }
  }

  const handleFeedSelect = (feedId: number) => {
    setSelectedFeed(feedId)
    setSelectedArticle(null)
    setSelectedActionFeed(null)
    setCurrentArticleDetail(null)

    if (feedId === 0) {
      // Special case for "All Articles" or "Newsfeed"
      loadRecentArticles()
    } else {
      loadFeedArticles(feedId)
    }
  }

  const handleArticleSelect = (articleId: number) => {
    setSelectedArticle(articleId)
    loadArticleDetail(articleId)
  }

  const handleActionFeedSelect = (action: string | null) => {
    // Update all state synchronously
    setSelectedActionFeed(action)
    setSelectedArticle(null)
    setSelectedFeed(null) // Set to null instead of 0
    setCurrentArticleDetail(null)

    if (action) {
      loadActionFeedArticles(action)
    } else {
      // If no action feed is selected, load recent articles
      loadRecentArticles()
    }
  }

  const handleBackToFeed = (shouldRefresh?: boolean) => {
    setSelectedArticle(null)
    setCurrentArticleDetail(null)

    // If refresh is requested, reload the current view
    if (selectedActionFeed) {
      // If viewing an action feed, reload it
      loadActionFeedArticles(selectedActionFeed)
    } else if (selectedFeed === 0) {
      // If viewing recent articles, reload them
      loadRecentArticles()
    } else if (selectedFeed) {
      // If viewing a specific feed, reload its articles
      loadFeedArticles(selectedFeed)
    }
  }

  const handleAddFeed = async (feed: FeedInfoSchema) => {
    try {
      // Add the new feed to the state
      const newExtendedFeed: ExtendedFeed = {
        ...feed,
        name: feed.title || `Feed ${feed.feed_id}`,
        count: 0,
        articles: [],
      }

      setFeeds((prevFeeds) => [...prevFeeds, newExtendedFeed])

      // Select the new feed
      setSelectedFeed(feed.feed_id)
      setSelectedArticle(null)
      setSelectedActionFeed(null)

      // Load articles for the new feed
      loadFeedArticles(feed.feed_id)
    } catch (error) {
      console.error("Failed to add feed:", error)
    }
  }

  const handleSidebarResize = (width: number) => {
    setSidebarWidth(width)
  }

  // Update the getDisplayTitle function to handle the new "similar" action
  const getDisplayTitle = () => {
    if (selectedActionFeed === "recommended") {
      return "Recommended Articles"
    }

    if (selectedActionFeed === "similar") {
      return "Similar Articles"
    }

    if (selectedActionFeed === "recent") {
      return "Recent Articles"
    }

    if (selectedActionFeed === "save") {
      return "Saved Articles"
    }

    if (selectedActionFeed === "upvote") {
      return "Upvoted Articles"
    }

    if (selectedFeed === 0) {
      return "Recent Articles"
    }

    if (selectedFeed && selectedFeed > 0) {
      const feed = feeds.find((f) => f.feed_id === selectedFeed)
      return feed?.title || feed?.name || ""
    }

    return "Recommended Articles" // Default fallback changed to recommended
  }

  // Update the handleRefreshCurrentView function to handle the new "similar" action
  const handleRefreshCurrentView = () => {
    if (selectedActionFeed === "recommended") {
      loadRecommendedArticles()
    } else if (selectedActionFeed === "similar") {
      loadSimilarArticles()
    } else if (selectedActionFeed === "recent") {
      loadRecentArticles()
    } else if (selectedActionFeed) {
      loadActionFeedArticles(selectedActionFeed)
    } else if (selectedFeed === 0) {
      loadRecentArticles()
    } else if (selectedFeed) {
      loadFeedArticles(selectedFeed)
    }
  }

  const handleFeedExpand = (feedId: number) => {
    const feed = feeds.find((f) => f.feed_id === feedId)
    // Only load articles if they haven't been loaded yet
    if (feed && feed.articles.length === 0) {
      loadFeedArticlesForSidebar(feedId)
    }
  }

  const handleArticleSkip = (articleId: number) => {
    // Remove article from current articles list
    setCurrentArticles((prevArticles) => prevArticles.filter((article) => article.article_id !== articleId))

    // Remove article from feeds sidebar if it exists there
    setFeeds((prevFeeds) =>
      prevFeeds.map((feed) => ({
        ...feed,
        articles: feed.articles.filter((article) => article.article_id !== articleId),
        count: Math.max(0, feed.count - (feed.articles.some((a) => a.article_id === articleId) ? 1 : 0)),
      })),
    )

    // If the skipped article was currently selected, clear the selection
    if (selectedArticle === articleId) {
      setSelectedArticle(null)
      setCurrentArticleDetail(null)
    }
  }

  return (
    <div className="flex min-h-screen bg-gray-100">
      <Sidebar
        feeds={feeds}
        selectedFeed={selectedFeed}
        selectedArticle={selectedArticle}
        selectedActionFeed={selectedActionFeed}
        onFeedSelect={handleFeedSelect}
        onArticleSelect={handleArticleSelect}
        onActionFeedSelect={handleActionFeedSelect}
        onAddFeed={handleAddFeed}
        onRefreshFeeds={refreshFeeds}
        onFeedExpand={handleFeedExpand}
        width={sidebarWidth}
        isLoading={isLoadingFeeds}
      />

      <ResizableDivider onResize={handleSidebarResize} initialWidth={sidebarWidth} minWidth={250} maxWidth={600} />

      <div className="flex-1 flex flex-col min-h-screen">
        {selectedArticle && currentArticleDetail ? (
          <ArticleDetailView
            article={currentArticleDetail}
            onBack={handleBackToFeed}
            isLoading={isLoadingArticleDetail}
          />
        ) : (
          <ArticleListView
            articles={currentArticles}
            feedName={getDisplayTitle()}
            onArticleSelect={handleArticleSelect}
            onArticleSkip={handleArticleSkip}
            onRefresh={handleRefreshCurrentView}
            isLoading={isLoadingArticles}
          />
        )}
      </div>
    </div>
  )
}

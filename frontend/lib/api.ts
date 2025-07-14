// API service functions for interacting with the HabitFeed backend

export interface ArticleSchema {
  article_id: number
  feed_id: number
  guid: string
  url: string
  title: string | null
  rss_description: string | null
  pub_date: string | null
  language: string | null
  full_text: string | null
}

export interface FeedInfoSchema {
  feed_id: number
  feed_url: string
  title: string | null
  last_checked: string | null
}

export interface FeedSchema extends FeedInfoSchema {
  articles: ArticleSchema[]
}

export interface FeedCreate {
  feed_url: string
}

export interface ActionRecord {
  action: string
  action_duration_ms?: number | null
}

export interface ActionResponse {
  message: string
  recorded_action: string
  article_id: number
}

export interface InteractionSchema {
  int_id: number
  article_id: number
  event_ts: string
  action_duration_ms?: number | null
  action: string
}

// Base API URL - replace with your actual API URL
const API_BASE_URL = "/api"

// Helper function for API requests
async function apiRequest<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`)
  }

  return response.json() as Promise<T>
}

// Get all feeds
export async function getAllFeeds(): Promise<FeedInfoSchema[]> {
  return apiRequest<FeedInfoSchema[]>("/feeds")
}

// Get a specific feed by ID
export async function getFeedById(feedId: number): Promise<FeedSchema> {
  return apiRequest<FeedSchema>(`/feeds/${feedId}`)
}

// Get articles for a specific feed
export async function getArticlesByFeed(feedId: number, limit?: number): Promise<ArticleSchema[]> {
  const queryParams = limit ? `?limit=${limit}` : ""
  return apiRequest<ArticleSchema[]>(`/feeds/${feedId}/articles${queryParams}`)
}

// Get recent articles across all feeds
export async function getRecentArticles(limit = 10): Promise<ArticleSchema[]> {
  return apiRequest<ArticleSchema[]>(`/articles/recent?limit=${limit}`)
}

// Get recommended articles across all feeds
export async function getRecommendedArticles(limit = 10): Promise<ArticleSchema[]> {
  return apiRequest<ArticleSchema[]>(`/articles/recommended-linucb?limit=${limit}`)
}

// Add a new function for similar articles (using the old recommended endpoint)
export async function getSimilarArticles(limit = 10): Promise<ArticleSchema[]> {
  return apiRequest<ArticleSchema[]>(`/articles/recommended?limit=${limit}`)
}

// Get a specific article by ID
export async function getArticleById(articleId: number): Promise<ArticleSchema> {
  return apiRequest<ArticleSchema>(`/articles/${articleId}`)
}

// Add a new feed
export async function addFeed(feedUrl: string): Promise<FeedInfoSchema> {
  return apiRequest<FeedInfoSchema>("/feeds", {
    method: "POST",
    body: JSON.stringify({ feed_url: feedUrl }),
  })
}

// Update all feeds
export async function updateAllFeeds(): Promise<void> {
  return apiRequest<void>("/feeds/update", {
    method: "POST",
  })
}

// Fetch all articles
export async function fetchAllArticles(): Promise<void> {
  return apiRequest<void>("/articles/fetch", {
    method: "POST",
  })
}

// Stream feed updates with progress
export function streamFeedUpdates(): string {
  return `${API_BASE_URL}/feeds/update/stream`
}

// Stream article fetching with progress
export function streamArticleFetch(): string {
  return `${API_BASE_URL}/articles/fetch/stream`
}

// Helper function to format relative time
export function formatRelativeTime(dateString: string | null): string {
  if (!dateString) return "Unknown"

  const date = new Date(dateString)
  const now = new Date()
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000)

  if (diffInSeconds < 60) return `${diffInSeconds}s`
  if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m`
  if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h`
  if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d`

  return date.toLocaleDateString()
}

// Helper function to extract domain from URL
export function extractSourceFromUrl(url: string): string {
  try {
    const domain = new URL(url).hostname.replace("www.", "")
    return domain
  } catch (e) {
    return "Unknown Source"
  }
}

// Record an action on an article
export async function recordArticleAction(
  articleId: number,
  action: string,
  durationMs?: number,
): Promise<ActionResponse> {
  const actionData: ActionRecord = {
    action,
    action_duration_ms: durationMs || null,
  }

  return apiRequest<ActionResponse>(`/articles/${articleId}/actions`, {
    method: "POST",
    body: JSON.stringify(actionData),
  })
}

// Get all actions for a specific article
export async function getArticleActions(articleId: number): Promise<InteractionSchema[]> {
  return apiRequest<InteractionSchema[]>(`/articles/${articleId}/actions`)
}

// Get articles by action type
export async function getArticlesByAction(action: string): Promise<ArticleSchema[]> {
  return apiRequest<ArticleSchema[]>(`/articles/action_feed/${action}`)
}

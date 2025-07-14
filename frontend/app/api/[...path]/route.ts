import { type NextRequest, NextResponse } from "next/server"

// Replace with your actual API URL
const API_BASE_URL = process.env.API_BASE_URL || "http://localhost:8000"

// Helper function to create a timeout signal
function createTimeoutSignal(timeoutMs: number): AbortSignal {
  const controller = new AbortController()
  setTimeout(() => controller.abort(), timeoutMs)
  return controller.signal
}

export async function GET(request: NextRequest, { params }: { params: { path: string[] } }) {
  const path = params.path.join("/")
  const searchParams = request.nextUrl.searchParams.toString()
  const url = `${API_BASE_URL}/${path}${searchParams ? `?${searchParams}` : ""}`

  try {
    // Check if this is an SSE endpoint
    if (path.includes("/stream")) {
      console.log("Attempting SSE connection to:", url)

      const response = await fetch(url, {
        headers: {
          Accept: "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          // Add ngrok-specific headers
          "ngrok-skip-browser-warning": "true",
        },
        // Add timeout for ngrok
        signal: createTimeoutSignal(30000), // 30 second timeout
      })

      if (!response.ok) {
        console.error(`SSE API returned ${response.status}: ${response.statusText}`)
        return NextResponse.json(
          { error: `API returned ${response.status}: ${response.statusText}` },
          { status: response.status },
        )
      }

      // For SSE, we need to stream the response
      return new NextResponse(response.body, {
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Headers": "Cache-Control",
        },
      })
    }

    // Regular JSON API request
    const response = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
        // Add ngrok-specific headers
        "ngrok-skip-browser-warning": "true",
      },
      // Add timeout
      signal: createTimeoutSignal(10000), // 10 second timeout
    })

    if (!response.ok) {
      console.error(`API returned ${response.status}: ${response.statusText}`)
      return NextResponse.json(
        { error: `API returned ${response.status}: ${response.statusText}` },
        { status: response.status },
      )
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error(`Error fetching ${url}:`, error)

    // More specific error messages
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        return NextResponse.json({ error: "Request timeout - ngrok connection may be slow" }, { status: 408 })
      }
      if (error.message.includes("fetch")) {
        return NextResponse.json(
          { error: "Failed to connect to API - check if ngrok tunnel is active" },
          { status: 503 },
        )
      }
    }

    return NextResponse.json(
      { error: "Failed to connect to API - please check if the API server is running" },
      { status: 500 },
    )
  }
}

export async function POST(request: NextRequest, { params }: { params: { path: string[] } }) {
  const path = params.path.join("/")
  const url = `${API_BASE_URL}/${path}`

  try {
    // Handle cases where request body might be empty
    let body = {}
    try {
      const requestText = await request.text()
      body = requestText ? JSON.parse(requestText) : {}
    } catch (e) {
      // If parsing fails, use empty object
      body = {}
    }

    const response = await fetch(url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        // Add ngrok-specific headers
        "ngrok-skip-browser-warning": "true",
      },
      body: JSON.stringify(body),
      // Add timeout
      signal: createTimeoutSignal(15000), // 15 second timeout
    })

    if (!response.ok) {
      console.error(`POST API returned ${response.status}: ${response.statusText}`)
      return NextResponse.json(
        { error: `API returned ${response.status}: ${response.statusText}` },
        { status: response.status },
      )
    }

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error(`Error posting to ${url}:`, error)

    // More specific error messages
    if (error instanceof Error) {
      if (error.name === "AbortError") {
        return NextResponse.json({ error: "Request timeout - ngrok connection may be slow" }, { status: 408 })
      }
    }

    return NextResponse.json(
      { error: "Failed to connect to API - please check if the API server and ngrok tunnel are running" },
      { status: 500 },
    )
  }
}

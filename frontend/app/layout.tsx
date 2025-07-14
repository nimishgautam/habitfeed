import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
      title: 'HabitFeed',
      description: 'A local RSS reader with AI-powered recommendations driven by your personal habits',
  generator: 'v0.dev',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}

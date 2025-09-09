import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Mic, Music, FileText, Users, Brain, User, Radio, Wifi } from 'lucide-react'
import UserSelector from '../UserSelector'
import ConnectionStatus from '../ConnectionStatus'

interface LayoutProps {
  children: React.ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navigationItems = [
    { path: '/audio', label: 'Audio Viewer', icon: Music },
    { path: '/annotation', label: 'Annotation', icon: FileText },
    { path: '/enrollment', label: 'Enrollment', icon: User },
    { path: '/speakers', label: 'Speakers', icon: Users },
    { path: '/inference', label: 'Inference', icon: Brain },
    { path: '/infer-live', label: 'Live Inference (Complex)', icon: Radio },
    { path: '/infer-live-simple', label: 'Live Inference', icon: Wifi },
  ]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <Mic className="h-8 w-8 text-blue-600" />
              <h1 className="text-xl font-semibold text-gray-900">
                Speaker Recognition System
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <ConnectionStatus />
              <div className="border-l border-gray-300 h-6"></div>
              <UserSelector />
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar Navigation */}
          <nav className="lg:w-64 flex-shrink-0">
            <div className="card p-4">
              <ul className="space-y-2">
                {navigationItems.map(({ path, label, icon: Icon }) => (
                  <li key={path}>
                    <Link
                      to={path}
                      className={`flex items-center space-x-3 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                        location.pathname === path
                          ? 'bg-blue-100 text-blue-900'
                          : 'text-gray-700 hover:bg-gray-100'
                      }`}
                    >
                      <Icon className="h-5 w-5" />
                      <span>{label}</span>
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </nav>

          {/* Main Content */}
          <main className="flex-1 min-w-0">
            <div className="card p-6">
              {children}
            </div>
          </main>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="text-center text-sm text-gray-500">
            🎤 Speaker Recognition System v0.1.0 | Built with React and PyTorch
          </div>
        </div>
      </footer>
    </div>
  )
}
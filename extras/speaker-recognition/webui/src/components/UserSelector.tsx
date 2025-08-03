import React, { useState } from 'react'
import { User, Plus, ChevronDown } from 'lucide-react'
import { useUser } from '../contexts/UserContext'

export default function UserSelector() {
  const { user, users, isLoading, selectUser, createUser } = useUser()
  const [isDropdownOpen, setIsDropdownOpen] = useState(false)
  const [isCreating, setIsCreating] = useState(false)
  const [newUsername, setNewUsername] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSelectUser = async (username: string) => {
    try {
      await selectUser(username)
      setIsDropdownOpen(false)
    } catch (error) {
      console.error('Failed to select user:', error)
    }
  }

  const handleCreateUser = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!newUsername.trim()) return

    setIsSubmitting(true)
    try {
      await createUser(newUsername.trim())
      setNewUsername('')
      setIsCreating(false)
      setIsDropdownOpen(false)
    } catch (error) {
      console.error('Failed to create user:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center space-x-2 text-gray-500">
        <User className="h-5 w-5" />
        <span>Loading...</span>
      </div>
    )
  }

  return (
    <div className="relative">
      {/* Current User Display / Dropdown Trigger */}
      <button
        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
        className="flex items-center space-x-2 px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
      >
        <User className="h-5 w-5 text-gray-600" />
        <span className="text-sm font-medium text-gray-900">
          {user ? user.username : 'Select User'}
        </span>
        <ChevronDown className="h-4 w-4 text-gray-500" />
      </button>

      {/* Dropdown Menu */}
      {isDropdownOpen && (
        <div className="absolute right-0 mt-2 w-64 bg-white rounded-md shadow-lg border z-50">
          <div className="py-2">
            {/* Existing Users */}
            {users.length > 0 && (
              <>
                <div className="px-3 py-1 text-xs font-medium text-gray-500 uppercase tracking-wide">
                  Select User
                </div>
                {users.map((u) => (
                  <button
                    key={u.id}
                    onClick={() => handleSelectUser(u.username)}
                    className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 transition-colors ${
                      user?.id === u.id ? 'bg-blue-50 text-blue-900' : 'text-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span>{u.username}</span>
                      {user?.id === u.id && (
                        <span className="text-xs text-blue-600">Current</span>
                      )}
                    </div>
                  </button>
                ))}
                <div className="border-t my-2"></div>
              </>
            )}

            {/* Create New User */}
            {!isCreating ? (
              <button
                onClick={() => setIsCreating(true)}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors flex items-center space-x-2"
              >
                <Plus className="h-4 w-4" />
                <span>Create New User</span>
              </button>
            ) : (
              <div className="px-3 py-2">
                <form onSubmit={handleCreateUser} className="space-y-2">
                  <input
                    type="text"
                    value={newUsername}
                    onChange={(e) => setNewUsername(e.target.value)}
                    placeholder="Username"
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                    autoFocus
                    disabled={isSubmitting}
                  />
                  <div className="flex space-x-2">
                    <button
                      type="submit"
                      disabled={!newUsername.trim() || isSubmitting}
                      className="flex-1 px-2 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      {isSubmitting ? 'Creating...' : 'Create'}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIsCreating(false)
                        setNewUsername('')
                      }}
                      className="flex-1 px-2 py-1 text-xs bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
                      disabled={isSubmitting}
                    >
                      Cancel
                    </button>
                  </div>
                </form>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Backdrop to close dropdown */}
      {isDropdownOpen && (
        <div
          className="fixed inset-0 z-40"
          onClick={() => setIsDropdownOpen(false)}
        />
      )}
    </div>
  )
}
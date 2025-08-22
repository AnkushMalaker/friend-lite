import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { ThemeProvider } from './contexts/ThemeContext'
import Layout from './components/layout/Layout'
import LoginPage from './pages/LoginPage'
import Chat from './pages/Chat'
import Conversations from './pages/Conversations'
import Memories from './pages/Memories'
import Users from './pages/Users'
import System from './pages/System'
import Upload from './pages/Upload'
import LiveRecord from './pages/LiveRecord'
import ProtectedRoute from './components/auth/ProtectedRoute'
import { ErrorBoundary, PageErrorBoundary } from './components/ErrorBoundary'

function App() {
  console.log('🚀 Full App restored with working login!')
  
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <AuthProvider>
          <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/" element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }>
                <Route index element={
                  <PageErrorBoundary>
                    <Conversations />
                  </PageErrorBoundary>
                } />
                <Route path="live-record" element={
                  <PageErrorBoundary>
                    <LiveRecord />
                  </PageErrorBoundary>
                } />
                <Route path="chat" element={
                  <PageErrorBoundary>
                    <Chat />
                  </PageErrorBoundary>
                } />
                <Route path="conversations" element={
                  <PageErrorBoundary>
                    <Conversations />
                  </PageErrorBoundary>
                } />
                <Route path="memories" element={
                  <PageErrorBoundary>
                    <Memories />
                  </PageErrorBoundary>
                } />
                <Route path="users" element={
                  <PageErrorBoundary>
                    <Users />
                  </PageErrorBoundary>
                } />
                <Route path="system" element={
                  <PageErrorBoundary>
                    <System />
                  </PageErrorBoundary>
                } />
                <Route path="upload" element={
                  <PageErrorBoundary>
                    <Upload />
                  </PageErrorBoundary>
                } />
              </Route>
            </Routes>
          </Router>
        </AuthProvider>
      </ThemeProvider>
    </ErrorBoundary>
  )
}

export default App
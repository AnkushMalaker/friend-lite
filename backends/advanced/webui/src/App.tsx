import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AuthProvider } from './contexts/AuthContext'
import { ThemeProvider } from './contexts/ThemeContext'
import Layout from './components/layout/Layout'
import LoginPage from './pages/LoginPage'
import Conversations from './pages/Conversations'
import Memories from './pages/Memories'
import Users from './pages/Users'
import System from './pages/System'
import Upload from './pages/Upload'
import ProtectedRoute from './components/auth/ProtectedRoute'

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/" element={
              <ProtectedRoute>
                <Layout>
                  <Routes>
                    <Route path="/" element={<Conversations />} />
                    <Route path="/conversations" element={<Conversations />} />
                    <Route path="/memories" element={<Memories />} />
                    <Route path="/users" element={<Users />} />
                    <Route path="/system" element={<System />} />
                    <Route path="/upload" element={<Upload />} />
                  </Routes>
                </Layout>
              </ProtectedRoute>
            } />
          </Routes>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  )
}

export default App
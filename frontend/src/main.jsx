import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

// Global fetch interceptor to redirect API calls directly to Render backend in production,
// bypassing Vercel's 4.5MB payload size limit for file uploads.
const apiBaseUrl = import.meta.env.VITE_API_URL || '';
if (apiBaseUrl) {
  const originalFetch = window.fetch;
  window.fetch = async (input, init) => {
    let url = input;
    if (typeof input === 'string' && input.startsWith('/api')) {
      url = apiBaseUrl + input;
    }
    return originalFetch(url, init);
  };
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <App />
)


import { useState } from 'react'
import './App.css'
import SearchIcon from './assets/mag.png'
import { Episode } from './types'

function App(): JSX.Element {
  const [searchTerm, setSearchTerm] = useState<string>('')
  const [episodes, setEpisodes] = useState<Episode[]>([])

  const handleSearch = async (value: string): Promise<void> => {
    setSearchTerm(value)
    
    if (value.trim() === '') {
      setEpisodes([])
      return
    }

    try {
      const response = await fetch(`/api/episodes?title=${encodeURIComponent(value)}`)
      const data: Episode[] = await response.json()
      setEpisodes(data)
    } catch (error) {
      console.error('Error fetching episodes:', error)
    }
  }

  return (
    <div className="full-body-container">
      <div className="top-text">
        <div className="google-colors">
          <h1 id="google-4">4</h1>
          <h1 id="google-3">3</h1>
          <h1 id="google-0-1">0</h1>
          <h1 id="google-0-2">0</h1>
        </div>
        <div className="input-box" onClick={() => document.getElementById('search-input')?.focus()}>
          <img src={SearchIcon} alt="search" />
          <input
            id="search-input"
            placeholder="Search for a Keeping up with the Kardashians episode"
            value={searchTerm}
            onChange={(e) => handleSearch(e.target.value)}
          />
        </div>
      </div>
      <div id="answer-box">
        {episodes.map((episode, index) => (
          <div key={index} className="episode-item">
            <h3 className="episode-title">{episode.title}</h3>
            <p className="episode-desc">{episode.descr}</p>
            <p className="episode-rating">IMDB Rating: {episode.imdb_rating}</p>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App

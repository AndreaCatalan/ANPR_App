import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [imageURL, setImageURL] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const imageRef = useRef()

  const uploadImage = (e) => {
    const { files } = e.target
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0])
      setImageURL(url)
      setResult(null)
      setError(null)
    }
  }

  const identifyImage = () => {
    if (!imageURL) return

    setLoading(true)
    setResult(null)
    setError(null)

    // simulate process
    setTimeout(() => {
      setLoading(false)

      // mock result (no backend yet so for UI for now)
      setResult({
        plate: 'ABC-1234',
        confidence: 0.94
      })
    }, 2000)
  }

  return (
    <div className="App">
      <h1>Automatic Number Plate Recognition System</h1>

      <div className="inputHolder">
        <input
          type="file"
          accept="image/*"
          className="uploadInput"
          onChange={uploadImage}
        />
      </div>

      <div className="mainWrapper">
        <div className="imageHolder">
          {imageURL ? (
            <img src={imageURL} alt="Preview" ref={imageRef} />
          ) : (
            <p className="placeholder">Upload an image to preview</p>
          )}
        </div>

        {imageURL && (
          <button
            className="button"
            onClick={identifyImage}
            disabled={loading}
          >
            {loading ? 'Processing...' : 'Identify Image'}
          </button>
        )}
      </div>

      {/* loading */}
      {loading && (
        <div className="loadingBox">
          <p>Analyzing image...</p>
        </div>
      )}

      {/* result */}
      {result && (
        <div className="resultBox">
          <h3>Recognition Result</h3>
          <p><strong>Plate Number:</strong> {result.plate}</p>
          <p><strong>Confidence:</strong> {(result.confidence * 100).toFixed(2)}%</p>
        </div>
      )}

      {/* error */}
      {error && (
        <div className="errorBox">
          {error}
        </div>
      )}
    </div>
  )
}

export default App

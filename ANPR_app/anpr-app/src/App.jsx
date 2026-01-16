import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [imageURL, setImageURL] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)

  const imageRef = useRef()

  const uploadImage = (e) => {
    const { files } = e.target
    if (files.length > 0) {
      const file = files[0]
      const url = URL.createObjectURL(file)
      setImageURL(url)
      setSelectedFile(file)
      setResult(null)
      setError(null)
    }
  }

  const identifyImage = async () => {
    if (!selectedFile) return

    setLoading(true)
    setResult(null)
    setError(null)

    try {
      // Create FormData to send file
      const formData = new FormData()
      formData.append('file', selectedFile)

      // Call backend API
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      
      if (data.success && data.detections && data.detections.length > 0) {
        // Use the first detection result
        const detection = data.detections[0]
        setResult({
          plateText: detection.plate_text,
          confidence: detection.confidence,
          ocrConfidence: detection.ocr_confidence,
          bbox: detection.bbox,
          plateImage: detection.plate_image,
          numCharacters: detection.num_characters,
          characterDetails: detection.character_details,
          isReadable: detection.is_readable
        })
      } else {
        setError('No license plates detected in the image')
      }
    } catch (err) {
      setError(`Error: ${err.message}. Make sure the backend server is running on http://localhost:8000`)
    } finally {
      setLoading(false)
    }
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
          <div className="resultGrid">
            <div className="resultItem">
              <strong>Plate Number:</strong> {result.plateText}
            </div>
            <div className="resultItem">
              <strong>Detection Confidence:</strong> {(result.confidence * 100).toFixed(2)}%
            </div>
            <div className="resultItem">
              <strong>OCR Confidence:</strong> {(result.ocrConfidence * 100).toFixed(2)}%
            </div>
            <div className="resultItem">
              <strong>Characters Detected:</strong> {result.numCharacters}
            </div>
            <div className="resultItem">
              <strong>Readable:</strong> {result.isReadable ? 'Yes' : 'No'}
            </div>
          </div>

          {/* Plate Image */}
          {result.plateImage && (
            <div className="plateImageContainer">
              <h4>Detected Plate:</h4>
              <img src={result.plateImage} alt="Detected Plate" className="plateImage" />
            </div>
          )}

          {/* Character Details */}
          {result.characterDetails && result.characterDetails.length > 0 && (
            <div className="characterDetails">
              <h4>Character Breakdown:</h4>
              <div className="characterGrid">
                {result.characterDetails.map((char, index) => (
                  <div key={index} className="characterItem">
                    <img src={char.image} alt={`Char ${index}`} className="charImage" />
                    <div className="charInfo">
                      <span className="charText">{char.character}</span>
                      <span className="charConfidence">{(char.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
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

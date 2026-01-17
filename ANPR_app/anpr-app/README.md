# ANPR App - Frontend

React-based frontend application for the Automatic Number Plate Recognition (ANPR) system. This application provides a user interface for uploading images and displaying license plate detection results.

## Features

- Image upload interface
- Real-time license plate detection
- Display of detection results including:
  - Detected plate number
  - Detection confidence
  - OCR confidence
  - Character breakdown with individual character recognition
  - Visual display of detected plate and characters

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn

## Installation

1. Navigate to the `anpr-app` directory:
   ```bash
   cd anpr-app
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

### Development Mode

Start the development server with hot-reload:
```bash
npm run dev
```

The application will be available at `http://localhost:5173` (or another port as shown in the terminal).

### Production Build

Build the application for production:
```bash
npm run build
```

Preview the production build:
```bash
npm run preview
```

## Configuration

The frontend connects to the backend API at `http://localhost:8000`. Make sure the backend server is running before using the application.

## Project Structure

```
anpr-app/
├── src/
│   ├── App.jsx          # Main application component
│   ├── App.css          # Application styles
│   ├── main.jsx         # Application entry point
│   └── index.css        # Global styles
├── public/              # Static assets
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
└── eslint.config.js     # ESLint configuration
```

## Dependencies

- **React 19.2.0**: UI library
- **React DOM 19.2.0**: React DOM renderer
- **Vite**: Build tool and dev server
- **ESLint**: Code linting

## API Endpoints Used

- `POST /detect` - Upload and process image for license plate detection
- `GET /health` - Check backend health status
- `GET /` - API information

## Troubleshooting

### Backend Connection Error
If you see "Make sure the backend server is running on http://localhost:8000":
1. Start the backend server (see anpr-backend README)
2. Verify the backend is accessible at `http://localhost:8000`
3. Check for CORS issues (backend should allow requests from the frontend)

### Port Conflicts
If port 5173 is already in use, Vite will automatically try the next available port. Check the terminal output for the correct URL.

### Image Upload Issues
- Ensure the uploaded file is a valid image format (JPEG, PNG, etc.)
- Check browser console for any errors
- Verify file size is reasonable (large files may cause timeouts)

## Development

### Code Style
The project uses ESLint with React-specific rules. Run the linter:
```bash
npm run lint
```

### Adding New Features
1. Create new components in the `src/` directory
2. Import and use them in `App.jsx` or create a new entry point
3. Add corresponding styles in `App.css` or create new CSS files

## License

This project is part of the ANPR system thesis project.

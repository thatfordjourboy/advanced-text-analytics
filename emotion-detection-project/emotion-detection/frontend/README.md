# Emotion Detection Frontend

React TypeScript frontend for the Emotion Detection ML Application.

## 🚀 Quick Start

### Install Dependencies
```bash
npm install
```

### Development
```bash
npm start
# or
npm run dev
```

### Build for Production
```bash
npm run build
```

### Testing
```bash
npm test
```

## 🛠️ Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **React Router** - Navigation
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **Framer Motion** - Animations

## 📁 Project Structure

```
src/
├── components/     # Reusable UI components
├── pages/         # Page components
├── hooks/         # Custom React hooks
├── services/      # API services
├── App.tsx        # Main app component
└── index.tsx      # App entry point
```

## 🔧 Configuration

- **Tailwind CSS** - Configured in `tailwind.config.js`
- **PostCSS** - Configured in `postcss.config.js`
- **TypeScript** - Configured in `tsconfig.json`
- **Proxy** - Set to `http://localhost:8000` for backend API

## 🌐 Environment Variables

Create a `.env` file in the root directory:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development
```

## 📦 Build Output

The build process creates a `build/` directory with optimized production files ready for deployment.

## 🚀 Deployment

This frontend can be deployed to:
- **Vercel** (recommended)
- **Netlify**
- **GitHub Pages**
- Any static hosting service

## 🔗 Backend Integration

The frontend connects to the FastAPI backend running on port 8000. Make sure the backend is running before testing the frontend.

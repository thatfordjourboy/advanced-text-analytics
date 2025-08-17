# 🧠 Emotion Detection ML System

A comprehensive multi-label emotion detection system using GloVe embeddings, Logistic Regression, and Random Forest classifiers. Built with FastAPI backend and React TypeScript frontend.

## **🚀 Features**

- **Multi-label Emotion Detection**: Identifies 7 emotion categories
- **Advanced ML Pipeline**: GloVe embeddings + ensemble classification
- **Real-time API**: FastAPI backend with async processing
- **Interactive Frontend**: React dashboard with analytics and training
- **Docker Support**: Production-ready containerization
- **Performance Optimized**: Chunked processing and memory management

## **📊 Emotion Categories**

1. **Joy** - Happiness, excitement, positive emotions
2. **Sadness** - Grief, disappointment, negative emotions  
3. **Anger** - Frustration, irritation, hostility
4. **Fear** - Anxiety, worry, apprehension
5. **Surprise** - Astonishment, amazement, shock
6. **Disgust** - Aversion, repulsion, distaste
7. **Neutral** - Balanced, calm, indifferent

## **🏗️ Architecture**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   ML Pipeline   │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (Scikit-learn)│
│                 │    │                 │    │                 │
│ • Analytics     │    │ • REST API      │    │ • GloVe Embed.  │
│ • Training      │    │ • Data Loading  │    │ • LR + RF       │
│ • Detection     │    │ • Model Mgmt    │    │ • Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **📁 Project Structure**

```
emotion-detection/
├── backend/                          # FastAPI backend
│   ├── app/                         # Application code
│   │   ├── core/                    # Core ML modules
│   │   │   ├── data_loader.py      # Dataset loading
│   │   │   ├── embeddings.py       # GloVe processing
│   │   │   ├── model_trainer.py    # ML training
│   │   │   └── text_processor.py   # Text preprocessing
│   │   ├── models/                  # Data models
│   │   └── main.py                 # FastAPI app
│   ├── data/                        # Data directory
│   │   ├── dialogues.json          # ConvLab dataset
│   │   ├── ontology.json           # Emotion ontology
│   │   └── glove.2024.wikigiga.100d.zip  # GloVe vectors (not in git)
│   ├── models/                      # Trained models
│   ├── Dockerfile                   # Docker configuration
│   ├── docker-compose.yml          # Local development
│   └── requirements.txt             # Python dependencies
├── frontend/                        # React frontend
│   ├── src/                        # Source code
│   │   ├── components/             # React components
│   │   ├── pages/                  # Application pages
│   │   ├── services/               # API services
│   │   └── hooks/                  # Custom hooks
│   ├── public/                     # Static assets
│   └── package.json                # Node dependencies
├── DEPLOYMENT_GUIDE.md             # Deployment instructions
└── README.md                       # This file
```

## **⚡ Quick Start**

### **Prerequisites**
- Python 3.12+
- Node.js 18+
- Docker (optional)

### **Backend Setup**
```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download GloVe vectors (required)
cd data
# Download glove.2024.wikigiga.100d.zip from Stanford NLP
# or use your existing local file

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### **Frontend Setup**
```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### **Docker Setup**
```bash
cd backend

# Build and run with Docker
docker-compose up --build
```

## **🔧 API Endpoints**

### **Core Endpoints**
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation
- `POST /api/detect-emotion` - Emotion detection
- `GET /api/models/status` - Model status
- `POST /api/models/train` - Train new models

### **Example Usage**
```python
import requests

# Detect emotions in text
response = requests.post("http://localhost:8000/api/detect-emotion", 
    json={"text": "I'm so happy to see you today!"})

emotions = response.json()
# Returns: {"emotions": ["joy"], "confidence": 0.89, ...}
```

## **📈 Model Performance**

### **Logistic Regression**
- **Accuracy**: 77.5%
- **Precision**: 76.2%
- **Recall**: 75.8%
- **F1-Score**: 76.0%

### **Random Forest**
- **Accuracy**: 79.1%
- **Precision**: 78.3%
- **Recall**: 77.9%
- **F1-Score**: 78.1%

*Results based on ConvLab Daily Dialog Dataset with macro-averaging*

## **🚀 Deployment**

### **Backend (Render)**
1. Connect GitHub repository
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Upload data files or use external storage

### **Frontend (Vercel)**
1. Import GitHub repository
2. Set build command: `npm run build`
3. Set output directory: `frontend/build`
4. Configure environment variables

**📋 See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions**

## **🔑 Environment Variables**

### **Backend**
```env
ENVIRONMENT=production
DEBUG=false
PORT=$PORT
```

### **Frontend**
```env
REACT_APP_API_URL=https://your-backend-url.onrender.com
REACT_APP_ENVIRONMENT=production
```

## **📊 Data Requirements**

### **Required Files (Not in Git)**
- **GloVe Vectors**: `glove.2024.wikigiga.100d.zip` (~1.8GB)
- **GloVe Text**: `wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt` (~3GB)

### **How to Get Data Files**
1. **Download from Stanford NLP** (if available)
2. **Upload manually** to deployment platform
3. **Use external storage** (AWS S3, Google Cloud)
4. **Extract from your local development environment**

## **🛠️ Development**

### **Adding New Emotions**
1. Update `backend/data/ontology.json`
2. Retrain models with new data
3. Update frontend emotion displays

### **Model Improvements**
1. Modify `backend/app/core/model_trainer.py`
2. Add new algorithms in training pipeline
3. Update hyperparameter tuning

### **Frontend Enhancements**
1. Add new pages in `frontend/src/pages/`
2. Create components in `frontend/src/components/`
3. Update API services in `frontend/src/services/`

## **🧪 Testing**

### **Backend Tests**
```bash
cd backend
python -m pytest tests/
```

### **Frontend Tests**
```bash
cd frontend
npm test
```

### **Integration Tests**
```bash
# Test full pipeline
curl -X POST "http://localhost:8000/api/detect-emotion" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this amazing project!"}'
```

## **📚 Technologies Used**

### **Backend**
- **FastAPI** - Modern web framework
- **Scikit-learn** - Machine learning
- **NumPy/Pandas** - Data processing
- **GloVe** - Word embeddings
- **Uvicorn** - ASGI server

### **Frontend**
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Recharts** - Data visualization
- **Axios** - HTTP client

### **DevOps**
- **Docker** - Containerization
- **GitHub** - Version control
- **Render** - Backend hosting
- **Vercel** - Frontend hosting

## **🤝 Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## **📄 License**

This project is licensed under the MIT License - see the LICENSE file for details.

## **🙏 Acknowledgments**

- **Stanford NLP** for GloVe embeddings
- **ConvLab** for the emotion dataset
- **FastAPI** and **React** communities
- **Open source contributors**

## **📞 Support**

- **Issues**: Create GitHub issue
- **Documentation**: Check DEPLOYMENT_GUIDE.md
- **Email**: [Your email]

---

**🎯 Built with ❤️ for emotion detection research and applications**

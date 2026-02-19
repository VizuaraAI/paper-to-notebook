# Paper2Notebook 📄→📓

Transform research papers into executable Jupyter notebooks in seconds. Powered by Gemini 2.5 Pro or local Ollama models.

![Paper2Notebook](https://img.shields.io/badge/AI-Powered-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![Next.js](https://img.shields.io/badge/next.js-14-black)

## 🎬 Demo

<video src="frontend/public/paper2notebook.mov" controls width="100%"></video>

## 🚀 Features

- **📄 PDF to Notebook Conversion**: Upload research papers and get runnable PyTorch notebooks
- **🔗 arXiv Integration**: Direct support for arXiv papers - just paste the URL
- **🤖 Dual LLM Support**: Use Gemini (cloud) or Ollama (local) for notebook generation
- **📐 LaTeX Extraction**: Automatically extracts and renders mathematical equations
- **🐍 PyTorch Implementation**: Real ML implementations at reduced scale for CPU execution
- **☁️ Google Colab Ready**: One-click "Open in Colab" functionality
- **🔧 Auto Dependencies**: Automatically detects and installs required libraries
- **🎯 Structured Output**: Organized into Abstract, Methodology, Experiments, and Conclusion sections

## 🛠️ Tech Stack

### Backend

- **FastAPI**: High-performance Python web framework
- **Gemini 2.5 Pro**: Cloud LLM for code generation
- **Ollama**: Local LLM inference (optional)
- **pypdf**: PDF text extraction for local LLM processing
- **nbformat**: Jupyter notebook generation

### Frontend

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Framer Motion**: Smooth animations
- **shadcn/ui**: Beautiful UI components

## 📋 Prerequisites

- Python 3.9+
- Node.js 18+
- **For Gemini**: API key ([Get one here](https://aistudio.google.com/apikey))
- **For Ollama**: Ollama installed locally or accessible on your network ([ollama.ai](https://ollama.ai))

## 🚀 Quick Start

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Add your Gemini API key and/or Ollama URL to .env

# Run the server
uvicorn app:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local

# Run the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📖 Usage

1. **Upload a Paper**: Drag and drop a PDF or paste an arXiv URL
2. **Select Provider**: Choose between Gemini (cloud) or Ollama (local)
3. **Enter API Key**: Provide your Gemini API key (only needed for Gemini provider)
4. **Generate**: Click generate and watch as the notebook is created
5. **Download or Open in Colab**: Get your executable notebook instantly

## 🏗️ Project Structure

```
paper-to-notebook/
├── backend/                 # FastAPI backend
│   ├── app.py              # Main application file
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment variables template
├── frontend/               # Next.js frontend
│   ├── app/               # App router pages
│   ├── components/        # React components
│   ├── public/           # Static assets
│   └── package.json      # Node dependencies
├── railway.toml          # Railway deployment config
└── README.md            # This file
```

## 🎯 How It Works

1. **PDF Processing**: Extracts text and structure from research papers
2. **AI Analysis**: The selected LLM (Gemini or Ollama) analyzes the paper's methodology and algorithms
3. **Code Generation**: Generates PyTorch implementation based on the paper
4. **Notebook Assembly**: Creates structured Jupyter notebook with:
   - Abstract and introduction
   - LaTeX equations
   - Python code cells
   - Experiment structure
   - Comments and documentation

## 🦙 Ollama Integration (Local LLM)

You can use **Ollama** instead of Gemini to run the entire pipeline locally — no cloud API key needed. This is useful if:

- You have Ollama installed on your own machine (`localhost`)
- You have an Ollama instance running on a machine in your office/home network (e.g. `192.168.x.x`)

### Setup

1. **Install Ollama** from [ollama.ai](https://ollama.ai) (on the machine that will run the models)

2. **Pull a model**:

   ```bash
   ollama pull mistral:7b
   ```

3. **Start the Ollama server**:

   ```bash
   ollama serve
   ```

4. **Configure the backend** — set `OLLAMA_BASE_URL` in `backend/.env`:

   ```bash
   # If Ollama is running on the same machine:
   OLLAMA_BASE_URL=http://localhost:11434

   # If Ollama is on another machine in your network:
   OLLAMA_BASE_URL=http://192.168.1.143:11434
   ```

5. **Select Ollama in the UI** — use the "Provider" dropdown to switch to "Ollama (Local)", then pick a model.


> **Note**: For Ollama to be accessible from another machine, make sure it is started with `OLLAMA_HOST=0.0.0.0:11434 ollama serve` so it binds to all network interfaces, not just localhost.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 📧 Contact

For any queries, mail us at: [raj.dandekar8@gmail.com](mailto:raj.dandekar8@gmail.com)

## 🔗 Links

- **GitHub**: [VizuaraAI/paper-to-notebook](https://github.com/VizuaraAI/paper-to-notebook)
- **Other Products**:
  - [Vizz-AI](https://vizz.vizuara.ai) - Personalized AI tutor
  - [Dynaroute](https://dynaroute.vizuara.ai) - Smart routing solution

---

**Note**: This tool is designed for educational and research purposes. Always verify generated code before use in production.

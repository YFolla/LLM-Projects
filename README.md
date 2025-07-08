# 🤖 LLM Engineering Projects

A comprehensive collection of practical Large Language Model applications built during my LLM Engineering course. Each project demonstrates different aspects of LLM integration, from basic text processing to complex multimodal applications with real-time data integration.

## 🎯 **Project Overview**

| Project | Description | Tech Stack | Key Features |
|---------|-------------|------------|--------------|
| **01 - Web Summarizer (BeautifulSoup)** | Extract and summarize web content using BeautifulSoup | Python, OpenAI API, BeautifulSoup, Requests | Web scraping, content extraction, AI summarization |
| **02 - Web Summarizer (Playwright)** | Advanced web scraping with JavaScript support | Python, OpenAI API, Playwright, Async processing | Dynamic content handling, modern web scraping |
| **03 - Ollama Summarizer** | Local LLM summarization using Ollama | Python, Ollama, Local LLM deployment | Privacy-focused, offline processing |
| **04 - Marketing Brochure Generator** | AI-powered marketing content creation | Python, OpenAI API, Gradio UI | Content generation, marketing automation |
| **05 - Coding Tutor** | Interactive programming learning assistant | Python, OpenAI API, Educational AI | Code explanation, learning assistance |
| **06 - Gradio Brochure Generator** | Web-based brochure creation tool | Python, Gradio, OpenAI API, Web UI | Interactive design, real-time generation |
| **07 - Gradio Chatbot** | Simple conversational AI interface | Python, Gradio, OpenAI API | Basic chat functionality, web interface |
| **08 - Airline Chatbot** | Flight booking assistant with real-time data | Python, OpenAI API, Amadeus SDK, Gradio | Flight search, real-time pricing, booking assistance |
| **09 - Multimodal Chatbot** | Advanced chatbot with voice, image, and flight capabilities | Python, OpenAI API, Whisper, DALL-E, Amadeus SDK | Voice transcription, image generation, multimodal interaction |
| **10 - Meeting Minutes Generator** | Automated meeting transcription and summarization | Python, OpenAI API, Audio processing | Meeting automation, transcription, summary generation |
| **11 - Code Converter** | Python to C++ code conversion with compilation | Python, Anthropic Claude, CodeQwen, Gradio, C++ compilation | Code translation, performance optimization, cross-platform compilation |
| **12 - RAG Pipeline** | Retrieval-Augmented Generation system with knowledge base | Python, LangChain, OpenAI API, ChromaDB, Gradio | Document retrieval, conversational memory, knowledge base querying |
| **13 - Frontier Fine-tuning** | GPT-4o-mini fine-tuning for price estimation | Python, OpenAI API, W&B, HuggingFace | Custom model training, price prediction, experiment tracking |
| **14 - Multi-Agent System** | Intelligent deal discovery with ensemble AI agents | Python, OpenAI API, Modal, ChromaDB, Gradio, Scikit-learn | Multi-agent coordination, ensemble modeling, real-time monitoring, automated deal hunting |

## 🛠️ **Technology Stack**

### **Core Technologies**
- **Python 3.8+** - Primary development language
- **OpenAI API** - GPT-4, Whisper, DALL-E integration
- **Anthropic Claude** - Advanced code generation
- **Gradio** - Web-based user interfaces
- **Hugging Face** - Model hosting and inference
- **LangChain** - RAG pipelines and document processing
- **Modal** - Cloud deployment for fine-tuned models

### **Specialized Libraries**
- **Web Scraping**: BeautifulSoup4, Playwright, Requests
- **Audio Processing**: Whisper API, Pydub, Audio transcription
- **Image Generation**: DALL-E 3, PIL, Base64 encoding
- **Travel APIs**: Amadeus SDK for flight data
- **Local LLMs**: Ollama integration
- **RAG & Vector Stores**: LangChain, ChromaDB, Document loaders
- **Compilation**: Subprocess, Platform detection, C++ toolchain
- **Model Training**: OpenAI Fine-tuning API, Weights & Biases
- **Machine Learning**: Scikit-learn, Random Forest, Linear Regression
- **Multi-Agent Systems**: Agent coordination, ensemble modeling

### **Development Tools**
- **Environment Management**: python-dotenv, Conda
- **API Integration**: REST APIs, SDK management
- **UI Framework**: Gradio for rapid prototyping
- **Experiment Tracking**: Weights & Biases (W&B)
- **Version Control**: Git, GitHub
- **Cloud Deployment**: Modal for scalable inference

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Python 3.8 or higher
python --version

# Install required packages
pip install -r requirements.txt
```

### **Environment Setup**
Create a `.env` file in the project root:
```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic Claude API Key
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Hugging Face Token
HF_TOKEN=your_hugging_face_token_here

# Amadeus API Credentials (for flight projects)
AMADEUS_API_KEY=your_amadeus_api_key_here
AMADEUS_API_SECRET=your_amadeus_api_secret_here

# Weights & Biases API Key (for fine-tuning projects)
WANDB_API_KEY=your_wandb_api_key_here

# Pushover API Keys (for multi-agent notifications)
PUSHOVER_USER_KEY=your_pushover_user_key_here
PUSHOVER_APP_TOKEN=your_pushover_app_token_here
```

### **Running Projects**
Each project is self-contained and can be run independently:
```bash
# Example: Run the multimodal chatbot
python 9_multimodal_chatbot.py

# Example: Run the code converter
python 11_code_coverter.py

# Example: Run the multi-agent system
cd "14_Multi_agent_system"
python gradio_interface.py
```

## 📋 **Project Details**

### **🌐 Web Processing Projects (1-3)**
- **Web Summarizers**: Demonstrate different approaches to web content extraction
- **Technologies**: BeautifulSoup vs Playwright for static vs dynamic content
- **Use Cases**: Content curation, research automation, news summarization

### **💼 Business Applications (4-6)**
- **Marketing Tools**: Automated content generation for business needs
- **Interactive Interfaces**: Gradio-based web applications
- **Use Cases**: Marketing automation, content creation, business productivity

### **🤖 Conversational AI (7-9)**
- **Progressive Complexity**: From basic chat to multimodal interactions
- **Advanced Features**: Voice processing, image generation, real-time data
- **Use Cases**: Customer service, travel planning, interactive assistance

### **⚙️ Productivity Tools (10-11)**
- **Meeting Automation**: Transcription and summarization workflows
- **Code Translation**: Python to C++ with optimization and compilation
- **Use Cases**: Business automation, performance optimization, cross-platform development

### **🔍 Knowledge Management (12)**
- **RAG Pipeline**: Retrieval-Augmented Generation with document knowledge base
- **Technologies**: LangChain, ChromaDB, OpenAI embeddings, conversational memory
- **Use Cases**: Document Q&A, knowledge base querying, enterprise search, intelligent assistance

### **🎯 Model Training & Fine-tuning (13)**
- **Frontier Fine-tuning**: Custom GPT-4o-mini training for specialized price estimation
- **Technologies**: OpenAI Fine-tuning API, Weights & Biases, data preprocessing
- **Features**: JSONL data preparation, experiment tracking, model evaluation, automated testing
- **Use Cases**: Domain-specific model adaptation, performance optimization, specialized task training

### **🤖 Multi-Agent Systems (14)**
- **Multi-Agent Deal Discovery**: Intelligent system with specialized AI agents for autonomous deal hunting
- **Technologies**: OpenAI API, Modal cloud deployment, ChromaDB, Scikit-learn ensemble models
- **Architecture**: Planning Agent orchestrates Scanner, Ensemble, and Messaging agents
- **Key Features**: 
  - **Ensemble Modeling**: Combines 3 pricing agents (Fine-tuned LLM, RAG, Random Forest)
  - **Real-time Monitoring**: Gradio web interface with live logging and auto-refresh
  - **Autonomous Operation**: Continuous RSS feed monitoring and deal analysis
  - **Smart Notifications**: Pushover integration for high-value deal alerts
  - **Persistent Memory**: Tracks discovered opportunities across sessions
- **Use Cases**: Automated deal hunting, e-commerce monitoring, pricing analysis, market research

## 🔧 **Technical Highlights**

### **Advanced Features**
- **Multimodal Processing**: Text, voice, and image integration
- **Real-time Data**: Live flight information and pricing
- **RAG Systems**: Document retrieval with conversational memory
- **Code Compilation**: Cross-platform C++ compilation with optimization
- **Async Processing**: Efficient handling of concurrent operations
- **Error Handling**: Robust error management and user feedback
- **Multi-Agent Coordination**: Intelligent agent orchestration and ensemble modeling
- **Cloud Deployment**: Modal-based scalable inference for fine-tuned models

### **Performance Optimizations**
- **Compiler Detection**: Automatic toolchain detection (GCC, Clang++, MSVC)
- **Platform Support**: Windows, macOS (Intel/Apple Silicon), Linux
- **Vector Embeddings**: Efficient document chunking and similarity search
- **Model Comparison**: Claude vs CodeQwen performance analysis
- **Optimization Flags**: M1/M2 specific compiler optimizations
- **Ensemble Learning**: Linear regression for optimal agent weight learning
- **Lazy Loading**: Efficient agent initialization and resource management

## 📚 **Learning Outcomes**

This collection demonstrates proficiency in:
- **LLM Integration**: Multiple AI providers and models
- **RAG Architecture**: Document processing, embeddings, and retrieval
- **Full-Stack Development**: Backend logic with frontend interfaces
- **API Management**: RESTful services and SDK management
- **Cross-Platform Development**: Multi-OS compatibility
- **Performance Engineering**: Code optimization and compilation
- **User Experience**: Intuitive interfaces and error handling
- **Multi-Agent Systems**: Agent coordination, ensemble modeling, and autonomous workflows
- **Cloud Deployment**: Scalable inference with Modal and distributed systems
- **Machine Learning**: Ensemble methods, model training, and evaluation

## 🔄 **Future Enhancements**

- [ ] Docker containerization for easy deployment
- [ ] CI/CD pipeline integration
- [ ] Additional LLM provider support
- [ ] Enhanced error handling and logging
- [ ] Performance benchmarking suite
- [ ] API rate limiting and caching
- [ ] Multi-language support
- [ ] Advanced multi-agent coordination patterns
- [ ] Real-time streaming interfaces
- [ ] Mobile-responsive web applications

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 **Contact**

For questions or collaboration opportunities, please reach out through GitHub issues or discussions.

---

**Built with ❤️ during LLM Engineering Course** 
# Multi-Agent Deal Discovery System

An intelligent multi-agent system that autonomously discovers, analyzes, and alerts on online deals using advanced AI techniques including ensemble modeling, RAG (Retrieval-Augmented Generation), and real-time web scraping.

## 🎯 **System Overview**

The system orchestrates multiple specialized AI agents to create a comprehensive deal-hunting pipeline:

1. **Scanner Agent** - Scrapes RSS feeds to discover new deals
2. **Ensemble Agent** - Combines 3 pricing models for accurate price estimation
3. **Messaging Agent** - Sends notifications via Pushover API
4. **Planning Agent** - Orchestrates the entire workflow

## 🏗️ **Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                    Gradio Web Interface                         │
│              (Real-time Monitoring & Control)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                Deal Agent Framework                             │
│           (Orchestration & Memory Management)                  │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  Planning Agent                                │
│              (Workflow Coordination)                           │
└─────┬───────────────┬───────────────────┬─────────────────────┘
      │               │                   │
┌─────▼─────┐  ┌─────▼─────┐      ┌─────▼─────┐
│  Scanner  │  │ Ensemble  │      │Messaging  │
│   Agent   │  │   Agent   │      │   Agent   │
└───────────┘  └─────┬─────┘      └───────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌───▼────┐  ┌────▼────┐
   │Specialist│  │Frontier│  │ Random  │
   │  Agent  │  │ Agent  │  │ Forest  │
   │(Modal)  │  │(RAG)   │  │ Agent   │
   └─────────┘  └────────┘  └─────────┘
```

## 🤖 **Agent Components**

### **Core Agents**

#### **Planning Agent** (`planning_agent.py`)
- **Role**: Orchestrates the entire deal discovery workflow
- **Responsibilities**: Coordinates Scanner, Ensemble, and Messaging agents
- **Threshold**: Only processes deals with >$30 potential savings
- **Logic**: Sorts deals by discount and selects the best opportunities

#### **Scanner Agent** (`scanner_agent.py`)
- **Role**: Discovers deals from RSS feeds and web scraping
- **Data Sources**: Multiple RSS feeds from deal websites
- **Features**: Duplicate detection, URL filtering, content extraction
- **Output**: Structured deal objects with product descriptions and prices

#### **Ensemble Agent** (`ensemble_agent.py`)
- **Role**: Combines predictions from 3 specialist pricing agents
- **Method**: Linear Regression model for optimal weight learning
- **Features**: Robust error handling, transparent logging
- **Performance**: Typically outperforms individual agents

#### **Messaging Agent** (`messaging_agent.py`)
- **Role**: Sends deal notifications via Pushover API
- **Features**: Rich notifications with product details and links
- **Configuration**: Requires Pushover user key and app token

### **Specialist Pricing Agents**

#### **Specialist Agent** (`specialist_agent.py`)
- **Technology**: Fine-tuned Llama 3.1 8B model on Modal cloud
- **Specialization**: Domain-specific price prediction
- **Features**: 4-bit quantization, PEFT adapters, cloud deployment
- **Performance**: Optimized for pricing-specific tasks

#### **Frontier Agent** (`frontier_agent.py`)
- **Technology**: OpenAI GPT-4o-mini with RAG
- **Data Source**: ChromaDB vector database with similar products
- **Features**: Sentence transformer embeddings, context-aware pricing
- **Strength**: Leverages similar product context for accuracy

#### **Random Forest Agent** (`random_forest_agent.py`)
- **Technology**: Scikit-learn Random Forest on product embeddings
- **Features**: Fast inference, robust to outliers
- **Training**: 100 estimators on sentence transformer vectors
- **Strength**: Consistent baseline performance

## 🌐 **Web Interface**

### **Gradio Interface** (`gradio_interface.py`)
- **Real-time Monitoring**: Live deal discovery with automatic refresh
- **Interactive Dashboard**: Click-to-select deals for notifications
- **Live Logging**: Real-time system logs with color-coded agents
- **Status Tracking**: System status, last update time, progress indicators
- **Responsive Design**: Clean, modern UI that adapts to screen sizes

### **Key Features**
- **Two-column Layout**: Deals table + live logs
- **Auto-refresh**: Updates every 60 seconds
- **Deal Selection**: Click any deal to send manual notification
- **Log Management**: Clear logs button and automatic log rotation
- **Status Panel**: Shows system state and last discovery cycle

## 📊 **Data Management**

### **Vector Database** (`vectorize.py`)
- **Technology**: ChromaDB with persistent storage
- **Embeddings**: Sentence transformer vectors for similarity search
- **Usage**: Powers RAG system for Frontier Agent
- **Setup**: Automated vectorization of product catalog

### **Persistent Memory** (`memory.json`)
- **Purpose**: Tracks discovered opportunities across sessions
- **Features**: Prevents duplicate notifications, maintains history
- **Format**: JSON serialization of Opportunity objects

### **Model Training**

#### **Ensemble Model** (`ensemble_model.py`)
- **Training Data**: 250 test samples from pricing dataset
- **Features**: Individual predictions + min/max derived features
- **Model**: Linear Regression with optimal weight learning
- **Output**: Saved model for production use

#### **Random Forest Model** (`random_forest.py`)
- **Training**: Product embeddings → price regression
- **Configuration**: 100 estimators, 8-core parallel training
- **Performance**: Fast inference with good baseline accuracy

## 🚀 **Setup Instructions**

### **Prerequisites**
```bash
# Python 3.8+
python --version

# Install dependencies
pip install scikit-learn pandas numpy joblib chromadb sentence-transformers 
pip install openai python-dotenv tqdm huggingface-hub gradio modal
```

### **Environment Configuration**
Create a `.env` file:
```env
# Required API Keys
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token

# Optional (for notifications)
PUSHOVER_USER_KEY=your_pushover_user_key
PUSHOVER_APP_TOKEN=your_pushover_app_token
```

### **Initial Setup**
```bash
# 1. Create vector database
python vectorize.py

# 2. Train Random Forest model
python random_forest.py

# 3. Train ensemble model
python ensemble_model.py

# 4. Launch web interface
python gradio_interface.py
```

## 💻 **Usage**

### **Web Interface**
```bash
# Start the Gradio interface
python gradio_interface.py

# Access at http://localhost:7860
# Interface will auto-refresh every 60 seconds
# Click any deal to send manual notification
```

### **Programmatic Usage**
```python
from deal_agent_framework import DealAgentFramework

# Initialize the framework
framework = DealAgentFramework()

# Run discovery cycle
opportunities = framework.run()

# Process results
for opp in opportunities:
    print(f"Deal: {opp.deal.product_description}")
    print(f"Price: ${opp.deal.price:.2f}")
    print(f"Estimate: ${opp.estimate:.2f}")
    print(f"Discount: ${opp.discount:.2f}")
```

### **Individual Agent Usage**
```python
import chromadb
from agents.ensemble_agent import EnsembleAgent

# Setup
client = chromadb.PersistentClient(path="products_vectorstore")
collection = client.get_or_create_collection('products')

# Initialize agent
agent = EnsembleAgent(collection)

# Get price prediction
price = agent.price("iPhone 15 Pro Max 256GB Space Black")
print(f"Predicted price: ${price:.2f}")
```

## 🧪 **Testing**

### **Comprehensive Test Suite** (`agent_tests/`)
- **Unit Tests**: 13 test methods with extensive mocking
- **Integration Tests**: End-to-end workflow validation
- **Live Data Tests**: Real API calls with cost controls

```bash
# Run mocked tests (fast, no API calls)
python agent_tests/test_planning_agent.py

# Run all tests including live data
RUN_LIVE_TESTS=1 python agent_tests/test_planning_agent.py
```

## 📈 **Performance Characteristics**

### **Ensemble Model**
- **Training Time**: Seconds on consumer hardware
- **Inference Speed**: <1 second per prediction
- **Accuracy**: Outperforms individual agents through ensemble averaging
- **Memory**: Minimal footprint with lazy loading

### **Web Interface**
- **Refresh Rate**: 60-second automatic updates
- **Responsiveness**: Real-time logging and status updates
- **Scalability**: Handles 100+ log entries efficiently
- **User Experience**: Clean, professional interface

### **Agent System**
- **Discovery Cycle**: 2-5 minutes per complete run
- **Deal Processing**: Up to 5 deals per cycle for efficiency
- **Threshold**: $30 minimum discount for notifications
- **Memory**: Persistent storage prevents duplicate processing

## 🔧 **Configuration**

### **System Parameters**
- **Deal Threshold**: $30 minimum discount (`planning_agent.py`)
- **Max Deals**: 5 deals processed per cycle
- **Refresh Rate**: 60 seconds for web interface
- **Log Retention**: 100 most recent entries

### **Model Paths**
- **Ensemble Model**: `ensemble_model.pkl`
- **Random Forest**: `random_forest_model.pkl`
- **Vector Database**: `products_vectorstore/`
- **Memory File**: `memory.json`

## 🛠️ **Troubleshooting**

### **Common Issues**

**Missing Models**:
```bash
# Retrain ensemble model
python ensemble_model.py

# Retrain Random Forest
python random_forest.py
```

**ChromaDB Issues**:
```bash
# Recreate vector database
python vectorize.py
```

**API Errors**:
- Check `.env` file for correct API keys
- Verify OpenAI API key has sufficient credits
- Ensure Modal service is deployed for Specialist Agent

**Interface Issues**:
- Check port 7860 is available
- Verify all dependencies are installed
- Check logs for initialization errors

## 🎯 **Use Cases**

- **Personal Deal Hunting**: Automated discovery of relevant deals
- **E-commerce Monitoring**: Track price changes and opportunities
- **Market Research**: Analyze pricing patterns and trends
- **Business Intelligence**: Competitive pricing analysis
- **Education**: Learn multi-agent system design patterns

## 🔮 **Future Enhancements**

- [ ] Additional RSS feed sources
- [ ] Machine learning model improvements
- [ ] Mobile-responsive interface enhancements
- [ ] API rate limiting and caching
- [ ] Docker containerization
- [ ] Multi-user support with personalization
- [ ] Advanced filtering and search capabilities

## 📄 **License**

This project is part of the LLM Engineering course and is licensed under the MIT License.

---

**Built with ❤️ using Python, OpenAI, ChromaDB, Gradio, and Modal** 
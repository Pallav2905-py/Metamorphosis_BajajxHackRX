# AWS Deployment Guide for BajajxHackRX Metamorphosis API

## Prerequisites
- AWS EC2 instance with Ubuntu/Amazon Linux
- Git repository already cloned
- SSH access to the server

## Step 1: Install Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+ and pip
sudo apt install python3 python3-pip python3-venv -y

# Install system dependencies for document processing
sudo apt install poppler-utils tesseract-ocr -y

# Navigate to your project directory
cd ~/Metamorphosis_BajajxHackRX

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Configure Environment Variables

```bash
# Create .env file
nano .env
```

Add your environment variables:
```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_index_name
GENAI_API_KEY=your_gemini_api_key
GENAI_MODEL=gemini-2.0-flash-exp
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_WORKERS=10
ENABLE_CACHE=true
CACHE_DIR=document_cache
BATCH_SIZE=32
TOP_K=15
CHUNK_OVERLAP=150
```

## Step 3: Configure AWS Security Group

In AWS Console:
1. Go to EC2 → Security Groups
2. Select your instance's security group
3. Add Inbound Rule:
   - Type: Custom TCP
   - Port Range: 8000
   - Source: 0.0.0.0/0 (for public access) or specific IPs
   - Description: FastAPI Application

## Step 4: Run the Application

### Option A: Direct Run (for testing)
```bash
# Activate virtual environment
source .venv/bin/activate

# Run the application
python app.py
```

### Option B: Production with Gunicorn
```bash
# Install Gunicorn
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app:app
```

### Option C: Background Process with nohup
```bash
# Run in background
nohup python app.py > app.log 2>&1 &

# Check if running
ps aux | grep python
```

## Step 5: Access Your API

Your API will be available at:
- `http://YOUR_EC2_PUBLIC_IP:8000`
- API docs: `http://YOUR_EC2_PUBLIC_IP:8000/docs`

## Step 6: Test the Deployment

```bash
# Test from your local machine
curl -X POST "http://YOUR_EC2_PUBLIC_IP:8000/api/v1/hackrx/run" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/document.pdf",
    "questions": ["What is this document about?"]
  }'
```

## Troubleshooting

### Check Application Logs
```bash
# If using nohup
tail -f app.log

# If using systemd service (see below)
sudo journalctl -u metamorphosis-api -f
```

### Check Port Status
```bash
# Check if port 8000 is open
sudo netstat -tlnp | grep 8000

# Check firewall (Ubuntu)
sudo ufw status
sudo ufw allow 8000
```

### Memory Issues
```bash
# Check memory usage
free -h
htop

# If low memory, consider adding swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

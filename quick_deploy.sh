#!/bin/bash

echo "🚀 Starting BajajxHackRX Metamorphosis API Deployment..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "🐍 Installing Python and system dependencies..."
sudo apt install python3 python3-pip python3-venv poppler-utils tesseract-ocr -y

# Setup virtual environment
echo "🔧 Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Create directories
echo "📁 Creating cache directory..."
mkdir -p document_cache

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  Please create .env file with your API keys!"
    echo "PINECONE_API_KEY=your_key_here" > .env
    echo "GENAI_API_KEY=your_key_here" >> .env
    echo "GENAI_MODEL=gemini-2.0-flash-exp" >> .env
    echo "📝 Edit .env file with your actual API keys before running the app"
else
    echo "✅ .env file found"
fi

# Make script executable
chmod +x quick_deploy.sh

echo "🎉 Deployment setup complete!"
echo "📝 Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Configure AWS Security Group (port 8000)"
echo "3. Run: source .venv/bin/activate && python app.py"
echo "4. Access: http://YOUR_EC2_IP:8000"

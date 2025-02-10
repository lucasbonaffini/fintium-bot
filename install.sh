#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Starting DexScreener Bot Installation..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories
mkdir -p logs
mkdir -p blacklists

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}.env file not found. Creating example .env file...${NC}"
    cp env.example .env
    echo -e "${GREEN}Please edit .env file with your configuration before running the bot.${NC}"
    exit 1
fi

# Pull and build containers
echo "Building Docker containers..."
docker-compose build

echo -e "${GREEN}Installation completed successfully!${NC}"
echo "To start the bot, run: docker-compose up -d"
echo "To view logs, run: docker-compose logs -f"
echo "To stop the bot, run: docker-compose down"
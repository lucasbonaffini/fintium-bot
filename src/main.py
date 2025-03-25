import asyncio
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
from trading_bot import TradingBot
import sys
import glob

async def setup_logging():
    """Configure logging with both file and console handlers"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = RotatingFileHandler(
        'trading_bot.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)

async def verify_environment():
    """Verify all required environment variables are set"""
    required_vars = [
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET',
        'DB_PASSWORD'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

async def main():
    bot = None
    try:
        # Load environment variables
        load_dotenv()
        
        # Configure logging
        await setup_logging()
        
        # Verify environment variables
        await verify_environment()
        
        logging.info("Starting trading bot...")
        
        # Create and initialize the bot
        bot = TradingBot()
        
        # Initialize components
        await bot.initialize()
        
        # Run the main loop
        await bot.run()
        
    except Exception as e:
        logging.critical(f"Fatal error in main: {str(e)}", exc_info=True)
        if bot:
            await bot.close()
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.critical(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)
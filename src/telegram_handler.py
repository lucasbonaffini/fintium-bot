from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging

class TelegramHandler:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.app = None
        self.trading_bot = None
        logging.info(f"Initializing TelegramHandler with chat_id: {chat_id}")

    def set_trading_bot(self, trading_bot):
        """Set reference to trading bot instance"""
        self.trading_bot = trading_bot
        logging.info("Trading bot reference set in TelegramHandler")

    async def initialize(self):
        """Initialize Telegram bot"""
        try:
            logging.info("Starting Telegram bot initialization...")
            self.app = Application.builder().token(self.token).build()
            
            # Register command handlers
            logging.info("Registering command handlers...")
            self.app.add_handler(CommandHandler("start", self.start_command))
            self.app.add_handler(CommandHandler("status", self.status_command))
            self.app.add_handler(CommandHandler("stats", self.stats_command))
            
            # Initialize and start bot
            logging.info("Starting Telegram bot application...")
            await self.app.initialize()
            await self.app.start()
            
            logging.info("Setting up bot commands...")
            await self.app.bot.set_my_commands([
                ('start', 'Start the bot and see available commands'),
                ('status', 'Check current bot status'),
                ('stats', 'View trading statistics')
            ])
                
            # Send test message
            await self.send_message("🤖 Bot initialized successfully!")
            logging.info("Telegram bot initialization completed")
            
        except Exception as e:
            logging.error(f"Error initializing Telegram bot: {str(e)}", exc_info=True)
            raise

    async def _verify_trading_bot(self) -> bool:
        """Verify trading bot is available"""
        if not self.trading_bot:
            logging.error("Trading bot reference not set")
            return False
        return True

    async def send_message(self, message: str):
        """Send message to configured chat"""
        try:
            if self.app and self.app.bot:
                logging.info(f"Sending message to chat {self.chat_id}: {message}")
                await self.app.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='HTML'
                )
                logging.info("Message sent successfully")
            else:
                logging.error("Cannot send message - bot not initialized")
        except Exception as e:
            logging.error(f"Error sending message: {str(e)}", exc_info=True)

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        try:
            user_id = update.effective_user.id
            logging.info(f"Received /start command from user {user_id}")
            await update.message.reply_text(
                "🚀 Trading Bot Started\n"
                "Available commands:\n"
                "/status - Check bot status\n"
                "/stats - View trading statistics"
            )
            logging.info("Start command response sent")
        except Exception as e:
            logging.error(f"Error in start command: {str(e)}", exc_info=True)
            await update.message.reply_text("Error processing command")

    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            user_id = update.effective_user.id
            logging.info(f"Received /status command from user {user_id}")
            
            if not await self._verify_trading_bot():
                await update.message.reply_text("Status check not available - bot not connected")
                return
                
            await self.trading_bot.check_market_status()
            logging.info("Status command processed")
        except Exception as e:
            logging.error(f"Error in status command: {str(e)}", exc_info=True)
            await update.message.reply_text("Error checking status")

    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        try:
            user_id = update.effective_user.id
            logging.info(f"Received /stats command from user {user_id}")
            
            if not await self._verify_trading_bot():
                await update.message.reply_text("Statistics not available - bot not connected")
                return
                
            stats = await self.trading_bot.get_bot_statistics()
            await update.message.reply_text(stats)
            logging.info("Stats command processed")
        except Exception as e:
            logging.error(f"Error in stats command: {str(e)}", exc_info=True)
            await update.message.reply_text("Error getting statistics")

    async def close(self):
        """Close Telegram bot"""
        try:
            if self.app:
                logging.info("Stopping Telegram bot...")
                await self.app.stop()
                await self.app.shutdown()
                logging.info("Telegram bot stopped")
        except Exception as e:
            logging.error(f"Error closing Telegram bot: {str(e)}", exc_info=True)
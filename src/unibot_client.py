from telethon import TelegramClient, events
import logging
import re
import asyncio
from datetime import datetime
from typing import Optional, Dict, Tuple
import os
import glob
from config import Config
from solscan_client import SolscanAPI

class UnibotSolanaClient:
    def __init__(self, config: Config, api_id: str, api_hash: str, phone: str):
        self.config = config
        self.api_id = api_id
        self.api_hash = api_hash
        self.phone = phone
        self.client = None
        self.unibot_username = 'UnisolUnibot'
        self.initialized = False
        self.command_queue = asyncio.Queue()
        self.is_processing = False
        self.session_name = f'unibot_solana_session_{api_id}'
        
        # Initialize Solscan client
        self.solscan = SolscanAPI(self.config.solscan)
        self.wallet_address = os.getenv('SOLANA_WALLET_ADDRESS', "GCczRdzhZYm5ZyEGae7eiZwdji833Ug9NYjkXmvtpbD2")
        
        # Configuration
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 30

    async def initialize(self):
        """Initialize client and connect to Unibot"""
        try:
            # Clean up old sessions
            await self._cleanup_old_sessions()

            # Initialize and start Telegram client
            self.client = TelegramClient(self.session_name, self.api_id, self.api_hash)
            await self.client.start(phone=self.phone)
            logging.info("Telegram client started successfully")
            
            # Connect to Unibot
            entity = await self.client.get_entity(self.unibot_username)
            logging.info(f"Connected to {entity.username}")
            
            # Start command processor
            if not self.is_processing:
                asyncio.create_task(self._process_command_queue())
            
            self.initialized = True
            
            
        except Exception as e:
            logging.error(f"Error initializing Unibot client: {str(e)}")
            await self.close()
            raise

    async def _cleanup_old_sessions(self):
        """Cleanup old session files"""
        try:
            session_files = glob.glob("*.session")
            for file in session_files:
                try:
                    os.remove(file)
                    logging.info(f"Removed old session file: {file}")
                except Exception as e:
                    logging.warning(f"Could not remove session file {file}: {e}")
        except Exception as e:
            logging.error(f"Error cleaning up sessions: {str(e)}")

    async def _process_command_queue(self):
        """Process commands in queue with rate limiting"""
        self.is_processing = True
        while True:
            try:
                command, future = await self.command_queue.get()
                
                if not self.initialized or not self.client:
                    if not future.done():
                        future.set_result((False, "Client not initialized"))
                    continue

                success, result = await self._execute_with_retry(command)
                if not future.done():
                    future.set_result((success, result))
                await asyncio.sleep(1)  # Rate limiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error processing command: {str(e)}")
            finally:
                self.command_queue.task_done()

    async def _execute_with_retry(self, command: str) -> Tuple[bool, str]:
        """Execute command with retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.client.send_message(self.unibot_username, command)
                # Wait for response or timeout
                await asyncio.sleep(2)
                return True, "Command sent successfully"
            except Exception as e:
                logging.error(f"Command attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        return False, "Command failed after all retries"

    async def _queue_command(self, command: str) -> Tuple[bool, str]:
        """Add command to queue and wait for result"""
        if not self.initialized:
            raise Exception("Client not initialized")
            
        future = asyncio.Future()
        await self.command_queue.put((command, future))
        return await future

    async def get_wallet_status(self) -> Dict:
        """Get comprehensive wallet status"""
        try:
            if not self.wallet_address:
                return {
                    'sol_balance': "0.0000 SOL",
                    'total_value_usd': "$0.00",
                    'timestamp': datetime.now()
                }

            # Get SOL balance using Solscan
            balance = await self.solscan.get_account_balance(self.wallet_address)
            
            if balance is not None:
                # Current SOL price - could be fetched from an API in production
                sol_price = 100.00  # Example fixed price
                sol_balance_str = f"{balance:.4f} SOL"
                total_value = f"${balance * sol_price:.2f}"
                
                logging.info(f"Retrieved wallet balance: {sol_balance_str}")
                
                return {
                    'sol_balance': sol_balance_str,
                    'total_value_usd': total_value,
                    'timestamp': datetime.now()
                }
            else:
                logging.error("Failed to get balance from Solscan")
                return {
                    'sol_balance': "Error getting balance",
                    'total_value_usd': "N/A",
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            logging.error(f"Error getting wallet status: {str(e)}")
            return {
                'sol_balance': "Error",
                'total_value_usd': "Error",
                'timestamp': datetime.now()
            }


    async def buy_token(self, token_address: str, amount: float, dex: str = 'raydium', slippage: float = 1.0) -> bool:
        """Execute buy order"""
        try:
            # Verify balance first
            if not await self.verify_balance_for_trade(amount):
                logging.error(f"Insufficient balance for trade of {amount} SOL")
                return False

            command = f"/buy {token_address} {amount}"
            success, _ = await self._queue_command(command)
            return success
            
        except Exception as e:
            logging.error(f"Error executing buy order: {str(e)}")
            return False

    async def sell_token(self, token_address: str, percentage: float = 100) -> bool:
        """Execute sell order"""
        try:
            command = f"/sell {token_address} {percentage}%"
            success, _ = await self._queue_command(command)
            return success
        except Exception as e:
            logging.error(f"Error executing sell order: {str(e)}")
            return False

    async def verify_balance_for_trade(self, amount_sol: float) -> bool:
        """Verify if balance is sufficient for trade"""
        try:
            if not self.wallet_address:
                logging.error("No wallet address available")
                return False
                
            balance = await self.solscan.get_account_balance(self.wallet_address)
            if balance is None:
                logging.error("Could not verify balance with Solscan")
                return False

            required_amount = amount_sol * 1.02  # 2% buffer for fees
            return balance >= required_amount
            
        except Exception as e:
            logging.error(f"Error verifying balance: {str(e)}")
            return False

    async def check_connection(self) -> bool:
        """Verify if connection is active"""
        try:
            if not self.initialized:
                return False
            success, _ = await self._queue_command("/wallet")
            return success
        except Exception as e:
            logging.error(f"Connection check failed: {str(e)}")
            return False

    async def close(self):
            """Close client connection and cleanup"""
            try:
                self.initialized = False
                if self.client:
                    await self.client.disconnect()
                    logging.info("Unibot client disconnected")
                
                # Close Solscan API session
                await self.solscan.close()
                
                # Clean up session file
                try:
                    session_file = f"{self.session_name}.session"
                    if os.path.exists(session_file):
                        os.remove(session_file)
                        logging.info("Session file cleared")
                except Exception as e:
                    logging.error(f"Error clearing session file: {str(e)}")
            except Exception as e:
                logging.error(f"Error closing client: {str(e)}")

    async def set_auto_sell(self, token_address: str, take_profit: float, stop_loss: float) -> bool:
        """Configure auto-sell parameters"""
        try:
            # Set take profit
            tp_success, _ = await self._queue_command(f"/tp {token_address} {take_profit}")
            if not tp_success:
                return False
                
            # Set stop loss
            sl_success, _ = await self._queue_command(f"/sl {token_address} {stop_loss}")
            return sl_success
            
        except Exception as e:
            logging.error(f"Error setting auto-sell parameters: {str(e)}")
            return False
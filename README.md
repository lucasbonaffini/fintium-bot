# Bitcoin Trading Bot with Moving Average Crossover Strategy

This is an automated cryptocurrency trading bot that implements a moving average crossover strategy for trading Bitcoin on Binance. The bot is built in Python and uses the CCXT library to interact with Binance's API.

## Features

- **Moving Average Crossover Strategy**: Uses fast and slow moving averages to identify potential entry and exit points
- **Binance API Integration**: Connects to Binance exchange for market data and trade execution
- **Database Storage**: Stores historical price data, trades, and performance metrics in PostgreSQL
- **Risk Management**: Implements position sizing based on account risk and volatility
- **Performance Tracking**: Calculates and reports key performance metrics like win rate, profit factor, etc.
- **Backtesting Capability**: Allows testing strategies on historical data before running with real money

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd fintium_bot
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up your PostgreSQL database.

5. Create a `.env` file based on `.env.example` and add your API keys:
   ```
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

## Configuration

The bot is configured through the `config.py` file, which includes:

- **Trading Strategy Parameters**: Moving average periods, timeframe, etc.
- **Risk Management Settings**: Position sizing, take profit, stop loss levels
- **Database Connection**: PostgreSQL database credentials
- **Binance API**: API key, secret, and connection settings

You can adjust these parameters to customize the bot's behavior.

## Usage

To start the trading bot:

```
python src/main.py
```

The bot will:
1. Connect to Binance and retrieve current market data
2. Apply the moving average crossover strategy to identify trading signals
3. Execute trades when signals are generated
4. Monitor open positions for exit conditions
5. Store all trade data and performance metrics in the database

## Strategy Details

The moving average crossover strategy works as follows:

- When the fast moving average crosses above the slow moving average, it generates a buy signal
- When the fast moving average crosses below the slow moving average, it generates a sell signal
- Additional filters like volume and RSI are applied to improve signal quality
- Take profit and stop loss levels are set for risk management

## Risk Warning

**IMPORTANT**: Trading cryptocurrencies involves significant risk and may not be suitable for everyone. The value of cryptocurrencies can go up or down, and you may lose some or all of your investment. This bot is provided for educational purposes only. Always start with small amounts and never trade with money you cannot afford to lose.

## License

[MIT License](LICENSE)

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
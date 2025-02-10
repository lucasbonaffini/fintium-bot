import re
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
from config import Config

class FilterManager:
    def __init__(self, config: Config):
        self.config = config
        self.load_blacklists()

    def load_blacklists(self):
        """Load blacklists from files"""
        try:
            for list_type, filepath in self.config.blacklists.blacklist_files.items():
                with open(filepath, 'r') as f:
                    addresses = {addr.strip().lower() for addr in f.readlines() if addr.strip()}
                    if list_type == 'tokens':
                        self.config.blacklists.blacklisted_tokens.update(addresses)
                    elif list_type == 'developers':
                        self.config.blacklists.blacklisted_developers.update(addresses)
                    elif list_type == 'contracts':
                        self.config.blacklists.blacklisted_contract_patterns.update(addresses)
        except FileNotFoundError as e:
            logging.warning(f"Blacklist file not found: {e}")

    def is_token_blacklisted(self, token_address: str) -> bool:
        """Check if token address is blacklisted"""
        return token_address.lower() in self.config.blacklists.blacklisted_tokens

    def is_developer_blacklisted(self, developer_address: str) -> bool:
        """Check if developer address is blacklisted"""
        return developer_address.lower() in self.config.blacklists.blacklisted_developers

    def has_suspicious_contract_pattern(self, contract_code: str) -> bool:
        """Check for suspicious patterns in contract code"""
        return any(
            re.search(pattern, contract_code, re.IGNORECASE)
            for pattern in self.config.blacklists.blacklisted_contract_patterns
        )

    def has_suspicious_name(self, token_name: str) -> bool:
        """Check if token name contains suspicious patterns"""
        name_lower = token_name.lower()
        if any(word in name_lower for word in self.config.filters.forbidden_names):
            return True
        return any(
            re.match(pattern, name_lower)
            for pattern in self.config.filters.suspicious_patterns
        )

    def check_token_metrics(self, token_data: Dict) -> tuple[bool, Optional[str]]:
        """
        Check if token meets minimum requirements
        Returns: (passes_check: bool, failure_reason: Optional[str])
        """
        # Check liquidity
        if token_data.get('liquidity', 0) < self.config.filters.min_liquidity:
            return False, "Insufficient liquidity"

        # Check market cap
        if token_data.get('marketCap', 0) < self.config.filters.min_market_cap:
            return False, "Market cap too low"

        # Check holders
        if token_data.get('holders', 0) < self.config.filters.min_holders:
            return False, "Too few holders"

        # Check max holder percentage
        if token_data.get('maxHolderPercentage', 100) > self.config.filters.max_holder_percentage:
            return False, "Whale concentration too high"

        # Check token age
        creation_time = datetime.fromtimestamp(token_data.get('creationTime', 0))
        min_age = datetime.now() - timedelta(hours=self.config.filters.min_age_hours)
        if creation_time > min_age:
            return False, "Token too new"

        return True, None
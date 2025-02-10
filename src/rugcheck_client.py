# rugcheck_client.py
import aiohttp
import logging
from typing import Dict, Optional, Tuple
from colorama import Fore, Style

class RugCheckClient:
    def __init__(self, security_manager):
        self.base_url = "https://api.rugcheck.xyz/v1/tokens"
        self.min_lp_locked_amount = 25000
        self.min_lp_locked_pct = 75
        self.max_holder_pct = 20
        self.security_manager = security_manager  # Agregar referencia al SecurityManager

    async def analyze_token_safety(self, token_address: str) -> tuple[bool, str]:
        """Complete token safety analysis with database updates"""
        try:
            report = await self.get_token_report(token_address)
            if not report:
                return False, "No se pudo obtener el reporte de RugCheck"

            # Check holders
            holders_safe, holder_message = self.check_top_holders(report.get('holders', []))
            if not holders_safe:
                # Actualizar blacklist si se detectan holders sospechosos
                await self.security_manager.add_to_blacklist(
                    token_address,
                    'coin',
                    f"Suspicious holder distribution: {holder_message}"
                )
                return False, holder_message

            # Check liquidity
            lp_safe, lp_message = self.check_liquidity_safety(report.get('markets', []))
            if not lp_safe:
                # Actualizar blacklist si hay problemas de liquidez
                await self.security_manager.add_to_blacklist(
                    token_address,
                    'coin',
                    f"Liquidity issues: {lp_message}"
                )
                return False, lp_message

            # Check risk score
            if report.get('riskScore', 0) > 50:
                await self.security_manager.add_to_blacklist(
                    token_address,
                    'coin',
                    f"High risk score: {report.get('riskScore')}"
                )
                return False, f"Risk score too high: {report.get('riskScore')}"

            # Check for malicious developers
            if report.get('creator'):
                await self.check_developer(report['creator'])

            return True, "Token passed all safety checks"

        except Exception as e:
            logging.error(f"Error in safety analysis: {str(e)}")
            return False, f"Error in analysis: {str(e)}"

    async def check_developer(self, developer_address: str):
        """Check and potentially blacklist developer"""
        try:
            dev_report = await self.get_developer_report(developer_address)
            if dev_report:
                if dev_report.get('rugPullCount', 0) > 0 or dev_report.get('suspiciousTokens', 0) > 2:
                    await self.security_manager.add_to_blacklist(
                        developer_address,
                        'dev',
                        f"Suspicious history: {dev_report.get('rugPullCount')} rug pulls, "
                        f"{dev_report.get('suspiciousTokens')} suspicious tokens"
                    )
        except Exception as e:
            logging.error(f"Error checking developer: {str(e)}")


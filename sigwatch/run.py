"""Entry point for the sigwatch trading bot."""
import asyncio
from bot.loop import run_bot

if __name__ == "__main__":
    asyncio.run(run_bot())

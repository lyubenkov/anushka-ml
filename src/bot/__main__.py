import asyncio
import logging
from .config import BotSettings
from .bot import setup_bot


async def main():
    # Load settings
    settings = BotSettings()
    bot = None  # Initialize bot as None

    # Setup logging
    logging.basicConfig(level=settings.log_level, format=settings.log_format)
    logger = logging.getLogger(__name__)

    # Create and start bot
    try:
        logger.info("Starting bot...")
        dp, bot = await setup_bot(token=settings.token, api_url=settings.api_url)

        # Start polling
        logger.info("Starting polling...")
        await dp.start_polling(bot)
    except Exception as e:
        logger.error(f"Bot failed to start: {e}")
        raise
    finally:
        logger.info("Bot stopped")
        if bot is not None:
            await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.error(f"Bot stopped due to error: {e}")

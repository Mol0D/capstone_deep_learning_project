import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler
from src.bot.handlers import BotHandlers
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()
TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')

# Setup logging
logger = setup_logger()

def main():
    # Create bot handlers
    handlers = BotHandlers()

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", handlers.start))
    application.add_handler(CommandHandler("help", handlers.help_command))
    application.add_handler(CommandHandler("test", handlers.test_command))
    application.add_handler(CommandHandler("predict_1h", handlers.predict_1h))
    application.add_handler(CommandHandler("predict_1d", handlers.predict_1d))

    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 
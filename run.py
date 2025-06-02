import sys
import time
import logging
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class BotReloader(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_bot()

    def start_bot(self):
        """Start the bot process"""
        if self.process:
            self.stop_bot()
        
        logger.info("Starting bot...")
        self.process = subprocess.Popen([sys.executable, 'bot.py'])
        logger.info("Bot started!")

    def stop_bot(self):
        """Stop the bot process"""
        if self.process:
            logger.info("Stopping bot...")
            self.process.terminate()
            self.process.wait()
            logger.info("Bot stopped!")
            self.process = None

    def on_modified(self, event):
        """Handle file modification events"""
        if event.src_path.endswith('.py'):
            logger.info(f"Detected change in {event.src_path}")
            self.start_bot()

def main():
    # Create the reloader
    reloader = BotReloader()
    
    # Set up the file system observer
    observer = Observer()
    observer.schedule(reloader, path='.', recursive=False)
    observer.start()
    
    logger.info("Watching for file changes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping file watcher...")
        observer.stop()
        reloader.stop_bot()
    
    observer.join()

if __name__ == '__main__':
    main() 
import os
import sys
import logging
from rq import Worker, Queue, Connection
import redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main worker process"""
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
    
    logger.info(f"Starting worker with Redis URL: {redis_url}")
    
    # Connect to Redis
    redis_conn = redis.from_url(redis_url)
    
    # Create queues
    queues = [
        Queue('novel_generation', connection=redis_conn),
        Queue('default', connection=redis_conn)
    ]
    
    logger.info(f"Worker listening on queues: {[q.name for q in queues]}")
    
    # Start worker
    with Connection(redis_conn):
        worker = Worker(queues)
        logger.info("Worker started successfully")
        worker.work()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.error(f"Worker failed: {e}")
        sys.exit(1)
import os
import sys

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import logger
from src.visualization import organize_figures

def main():
    logger.info("================== GENERATING PUBLICATION FIGURES ==================")
    organize_figures()
    logger.info("Publication figures generated and organized.")

if __name__ == "__main__":
    main()

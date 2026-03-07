import os
import zipfile
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """ 
    Runs scripts to download competition data from Kaggle 
    and extract it into the raw data directory.
    """
    parser = argparse.ArgumentParser(description="Download Kaggle competition data.")
    parser.add_argument(
        'competition_name', 
        type=str, 
        help="The Kaggle competition name (e.g. 'spaceship-titanic' or 'titanic')"
    )
    args = parser.parse_args()

    # Find the project root directory
    project_dir = Path(__file__).resolve().parents[2]
    
    # Load environment variables from .env file
    # This must be done BEFORE importing kaggle since the kaggle package 
    # reads environment variables on import / when authenticating.
    env_path = project_dir / '.env'
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        load_dotenv(dotenv_path=env_path)
    else:
        logger.warning("No .env file found. Make sure KAGGLE_API_TOKEN is set.")

    # Import kaggle after dotenv is loaded
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    # Initialize and authenticate the Kaggle API
    api = KaggleApi()
    try:
        api.authenticate()
        logger.info("Successfully authenticated with Kaggle API.")
    except Exception as e:
        logger.error(f"Failed to authenticate with Kaggle API: {e}")
        logger.error("Please make sure your KAGGLE_API_TOKEN is correctly set in the .env file.")
        return

    raw_data_dir = project_dir / "data" / "raw"
    
    # Ensure data/raw directory exists
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Competition name from arguments
    competition_name = args.competition_name
    
    logger.info(f"Downloading '{competition_name}' dataset to {raw_data_dir}...")
    
    try:
        # Download the competition files
        api.competition_download_files(competition_name, path=raw_data_dir)
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg or "Forbidden" in error_msg:
            logger.error(f"Failed to download the dataset. You probably haven't accepted the competition rules.")
            logger.error(f"Please go to https://www.kaggle.com/c/{competition_name}/rules to accept them.")
        else:
            logger.error(f"Failed to download the dataset: {e}")
        return
    
    # Unzip the downloaded files
    zip_path = raw_data_dir / f"{competition_name}.zip"
    if zip_path.exists():
        logger.info(f"Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
            
            # Remove the zip file after extraction
            zip_path.unlink()
            logger.info("Extraction complete. Raw data is ready.")
        except zipfile.BadZipFile:
            logger.error(f"The downloaded file {zip_path.name} is not a valid zip file.")
    else:
        logger.error("Download failed or zip file not found.")

if __name__ == '__main__':
    main()
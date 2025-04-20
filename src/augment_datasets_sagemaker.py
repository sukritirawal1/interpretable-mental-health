#!/usr/bin/env python
# coding: utf-8

"""
Script for augmenting mental health datasets using techniques from ICON 2021 paper.
Designed to run on AWS SageMaker for handling the computationally intensive tasks.

Usage:
    python augment_datasets_sagemaker.py --input_dir /path/to/input --output_dir /path/to/output
"""

import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import time
import torch

# Import the data augmenter
from data_augmentation import MentalHealthAugmenter, augment_selected_datasets

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_sagemaker_paths(is_sagemaker=False):
    """
    Set up paths for SageMaker or local execution.
    
    Args:
        is_sagemaker: Whether running in SageMaker environment
        
    Returns:
        Dict of paths
    """
    if is_sagemaker:
        # SageMaker paths
        paths = {
            'data_dir': '/opt/ml/input/data',
            'output_dir': '/opt/ml/output/data',
            'model_dir': '/opt/ml/model',
        }
    else:
        # Local paths - using paths relative to the MentalLLaMA repository
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        paths = {
            'data_dir': os.path.join(base_dir, '..', 'MentalLLaMA-main', 'train_data'),
            'output_dir': os.path.join(base_dir, 'augmented_data'),
            'model_dir': os.path.join(base_dir, 'models'),
        }
    
    # Create directories if they don't exist
    for dir_path in paths.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return paths

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Augment mental health datasets for MentalLLaMA using techniques from ICON 2021 paper"
    )
    
    parser.add_argument('--input_dir', type=str, 
                        help="Directory containing datasets to augment")
    parser.add_argument('--output_dir', type=str, 
                        help="Directory to save augmented datasets")
    parser.add_argument('--datasets', nargs='+', default=['DR', 'dreaddit', 'SAD'],
                        help="Dataset names to augment (default: DR, dreaddit, SAD)")
    parser.add_argument('--augmentation_ratio', type=int, default=1,
                        help="Number of augmented samples per original sample (default: 1)")
    parser.add_argument('--sagemaker', action='store_true',
                        help="Running in SageMaker environment")
    parser.add_argument('--max_samples', type=int, default=None,
                        help="Maximum number of samples to process per dataset (for testing)")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up paths
    paths = setup_sagemaker_paths(args.sagemaker)
    
    # Override with command line arguments if provided
    input_dir = args.input_dir or os.path.join(paths['data_dir'], 'complete_data')
    output_dir = args.output_dir or paths['output_dir']
    
    # Log environment info
    logger.info(f"Python version: {os.sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Log parameters
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Datasets to augment: {args.datasets}")
    logger.info(f"Augmentation ratio: {args.augmentation_ratio}")
    if args.max_samples:
        logger.info(f"Testing with max {args.max_samples} samples per dataset")
    
    # Start timer
    start_time = time.time()
    
    try:
        # Perform augmentation
        logger.info("Starting dataset augmentation...")
        augment_selected_datasets(
            input_dir,
            output_dir,
            args.datasets,
            args.augmentation_ratio,
            args.max_samples
        )
        logger.info("Dataset augmentation completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during augmentation: {e}")
        raise
    
    # End timer
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Total processing time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Generate a summary report
    try:
        summary = {"dataset": [], "original_samples": [], "augmented_samples": []}
        for dataset in args.datasets:
            # Look for augmented files
            for root, _, files in os.walk(output_dir):
                for file in files:
                    if dataset.lower() in file.lower() and file.endswith('.csv'):
                        file_path = os.path.join(root, file)
                        df = pd.read_csv(file_path)
                        
                        # Count original vs augmented
                        if 'is_augmented' in df.columns:
                            original = df[df['is_augmented'] == 0].shape[0]
                            augmented = df[df['is_augmented'] == 1].shape[0]
                        else:
                            # Estimate based on ratio
                            total = df.shape[0]
                            original = total // (args.augmentation_ratio + 1)
                            augmented = total - original
                        
                        summary["dataset"].append(file)
                        summary["original_samples"].append(original)
                        summary["augmented_samples"].append(augmented)
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary)
        summary_path = os.path.join(output_dir, "augmentation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary saved to {summary_path}")
        
        # Log the summary
        for i, dataset in enumerate(summary["dataset"]):
            logger.info(f"{dataset}: {summary['original_samples'][i]} original + "
                        f"{summary['augmented_samples'][i]} augmented = "
                        f"{summary['original_samples'][i] + summary['augmented_samples'][i]} total")
        
    except Exception as e:
        logger.warning(f"Could not generate summary: {e}")

if __name__ == '__main__':
    main()
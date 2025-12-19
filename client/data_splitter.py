#!/usr/bin/env python3
"""
Data splitting script for heart disease dataset.
Divides raw data into train, validation, and test sets.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from common.constants import DATA_PATHS, RANDOM_SEED, TARGET_COLUMN
from common.serialization import save_dataframe


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSplitter:
    """Handles data splitting into train/val/test sets."""
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        random_state: int = RANDOM_SEED,
        stratify: bool = True
    ):
        """
        Initialize DataSplitter.
        
        Args:
            train_size: Proportion for training set (default 0.7)
            val_size: Proportion for validation set (default 0.15)
            test_size: Proportion for test set (default 0.15)
            random_state: Random seed for reproducibility
            stratify: Whether to stratify split by target variable
        """
        # Validate sizes
        total = train_size + val_size + test_size
        if not (0.99 < total < 1.01):
            raise ValueError(
                f"Train + val + test sizes must sum to 1.0, got {total}"
            )
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        
    def split(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into train, validation, and test sets.
        
        Args:
            df: Input dataframe
            target_col: Name of target column for stratification
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Starting data split. Total samples: {len(df)}")
        
        # Prepare stratification
        stratify_col = df[target_col] if self.stratify else None
        
        # First split: train vs temp (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=self.train_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        # Second split: val vs test from temp
        val_test_ratio = self.val_size / (self.val_size + self.test_size)
        
        if self.stratify:
            stratify_col = temp_df[target_col]
        
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_test_ratio,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        logger.info(f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
        logger.info(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
        logger.info(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    @staticmethod
    def check_distribution(
        original_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str
    ) -> None:
        """
        Check target variable distribution across splits.
        
        Args:
            original_df: Original dataframe
            train_df: Training set
            val_df: Validation set
            test_df: Test set
            target_col: Target column name
        """
        logger.info("\n" + "="*60)
        logger.info("TARGET VARIABLE DISTRIBUTION")
        logger.info("="*60)
        
        for name, df in [("Original", original_df), ("Train", train_df), 
                         ("Validation", val_df), ("Test", test_df)]:
            dist = df[target_col].value_counts(normalize=True) * 100
            logger.info(f"\n{name}:")
            for class_label, pct in dist.items():
                logger.info(f"  Class {class_label}: {pct:.2f}%")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Split heart disease dataset into train/val/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default split (70/15/15)
  python client/data_splitter.py
  
  # Custom split
  python client/data_splitter.py --input data/raw/heart_disease.csv --train-size 0.8 --val-size 0.1 --test-size 0.1
  
  # Without stratification
  python client/data_splitter.py --no-stratify
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DATA_PATHS["raw"],
        help="Path to raw data CSV file (default: data/raw/heart_disease.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DATA_PATHS["processed"],
        help="Directory to save processed data splits (default: data/processed)"
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Proportion for training set (default: 0.7)"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion for validation set (default: 0.15)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion for test set (default: 0.15)"
    )
    parser.add_argument(
        "--no-stratify",
        action="store_true",
        help="Disable stratified splitting (random split)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED})"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=TARGET_COLUMN,
        help=f"Target column name for stratification (default: {TARGET_COLUMN})"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f" Input file not found: {input_path}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f" Output directory: {output_dir}")
    
    try:
        # Load data
        logger.info(f" Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f" Loaded {len(df)} samples with {len(df.columns)} features")
        
        # Validate target column exists
        if args.target_col not in df.columns:
            logger.error(f" Target column '{args.target_col}' not found in data")
            logger.info(f"Available columns: {', '.join(df.columns)}")
            return 1
        
        # Initialize splitter
        splitter = DataSplitter(
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_seed,
            stratify=not args.no_stratify
        )
        
        # Perform split
        logger.info(f" Splitting data ({args.train_size*100:.0f}/{args.val_size*100:.0f}/{args.test_size*100:.0f})...")
        train_df, val_df, test_df = splitter.split(df, args.target_col)
        
        # Check distribution
        splitter.check_distribution(df, train_df, val_df, test_df, args.target_col)
        
        # Save splits
        logger.info("\nSaving data splits...")
        save_dataframe(train_df, output_dir / "train.csv")
        save_dataframe(val_df, output_dir / "validation.csv")
        save_dataframe(test_df, output_dir / "test.csv")
        
        logger.info("\n" + "="*60)
        logger.info(" Data splitting completed successfully!")
        logger.info("="*60)
        logger.info(f"Train:      {output_dir}/train.csv ({len(train_df)} samples)")
        logger.info(f"Validation: {output_dir}/validation.csv ({len(val_df)} samples)")
        logger.info(f"Test:       {output_dir}/test.csv ({len(test_df)} samples)")
        
        return 0
        
    except Exception as e:
        logger.error(f" Error during data splitting: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
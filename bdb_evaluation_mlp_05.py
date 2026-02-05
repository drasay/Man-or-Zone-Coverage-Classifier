import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def clean_frame_ids(predictions):
    """
    Clean frameUniqueId and extract proper frameId and uniqueId, handling augmented frames.
    
    Parameters:
    predictions: DataFrame with frameUniqueId column
    
    Returns:
    DataFrame: Updated predictions with proper IDs
    """
    # Copy the dataframe to avoid modifying the original
    preds = predictions.copy()
    
    # Initialize columns if they don't exist
    if 'uniqueId' not in preds.columns:
        preds['uniqueId'] = ""
    if 'frameId' not in preds.columns:
        preds['frameId'] = -1
        
    # Add is_augmented column
    preds['is_augmented'] = False
    
    # Process each row to handle regular and augmented frames
    for idx, row in preds.iterrows():
        frame_id = row['frameUniqueId']
        
        # Handle augmented frames
        base_frame_id = frame_id.replace("_aug", "")
        is_aug = "_aug" in frame_id

        preds.at[idx, 'is_augmented'] = is_aug
        preds.at[idx, 'uniqueId'] = "_".join(base_frame_id.split("_")[:-1])
        preds.at[idx, 'frameId'] = int(base_frame_id.split("_")[-1])
        preds.at[idx, 'clean_frameUniqueId'] = base_frame_id
    
    return preds

def process_and_visualize(week_number):
    """
    Complete workflow function to process predictions and visualize accuracy,
    handling both regular and augmented frames.
    
    Parameters:
    week_number: The week to process
    
    Returns:
    processed_predictions: The processed prediction dataframe
    accuracy_data: The accuracy by frame dataframe
    """
    # Read the predictions file
    pred_file = f"C:/python/nfl-big-data-bowl-2025/week_{week_number}_preds_mlp.csv"
    print(f"Reading predictions from {pred_file}")
    predictions = pd.read_csv(pred_file)
    
    # Clean frame IDs and handle augmented frames
    print("Processing frame IDs and handling augmented frames...")
    predictions = clean_frame_ids(predictions)
    
    # Read tracking data to get frames_from_snap
    tracking_file = f"C:/python/nfl-big-data-bowl-2025/tracking_week_{week_number}.csv"
    print(f"Reading tracking data from {tracking_file}")
    
    # Only read necessary columns from tracking data to improve performance
    tracking_data = pd.read_csv(tracking_file, 
                              usecols=['gameId', 'playId', 'frameId', 'frameType'])
    
    # Create IDs for joining
    tracking_data['uniqueId'] = tracking_data['gameId'].astype(str) + "_" + tracking_data['playId'].astype(str)
    tracking_data['frameUniqueId'] = (
        tracking_data['gameId'].astype(str) + "_" + 
        tracking_data['playId'].astype(str) + "_" + 
        tracking_data['frameId'].astype(str)
    )
    
    # Extract snap frames
    print("Finding snap frames...")
    snap_frames = tracking_data[tracking_data['frameType'] == 'SNAP'].groupby('uniqueId')['frameId'].first()
    
    # Create a mapping from uniqueId to snap_frame
    snap_frame_dict = snap_frames.to_dict()
    
    # Apply the mapping to get snap_frame for each row in predictions
    predictions['snap_frame'] = predictions['uniqueId'].map(snap_frame_dict)
    
    # Calculate frames_from_snap
    predictions['frames_from_snap'] = predictions['frameId'] - predictions['snap_frame']
    
    # Convert to seconds
    predictions['seconds_from_snap'] = predictions['frames_from_snap'] / 10
    
    # Add correctness flag
    predictions['base_correct'] = (predictions['pred'] == predictions['actual']).astype(int)
    
    # Check for missing snap frames
    missing_snaps = predictions['snap_frame'].isna().sum()
    if missing_snaps > 0:
        print(f"Warning: {missing_snaps} rows ({missing_snaps/len(predictions):.1%}) have missing snap frames")
        
        # List uniqueIds with missing snaps for debugging
        missing_ids = predictions[predictions['snap_frame'].isna()]['uniqueId'].unique()
        print(f"First few uniqueIds with missing snaps: {missing_ids[:5]}")
        
        # Drop rows with missing snap frames for analysis
        predictions = predictions.dropna(subset=['snap_frame'])
    
    # Filter to relevant window
    filtered = predictions[(predictions['frames_from_snap'] < 15) & 
                         (predictions['frames_from_snap'] > -100)]
    
    # Check if we have enough data
    if len(filtered) < 10:
        print(f"Warning: Only {len(filtered)} data points after filtering. Check your data.")
        return predictions, None
    
    # Ensure is_augmented column exists
    if 'is_augmented' not in filtered.columns:
        filtered['is_augmented'] = False
    
    # Visualize the results
    accuracy_data = visualize_accuracy_by_frame(filtered)
    
    return predictions, accuracy_data

def visualize_accuracy_by_frame(filtered_preds):
    """
    Create an accurate visualization of frame-by-frame accuracy.
    
    Parameters:
    filtered_preds: DataFrame with 'frames_from_snap', 'seconds_from_snap', and 'base_correct' columns
    
    Returns:
    DataFrame: Accuracy by frame data
    """
    filtered_preds = filtered_preds.copy()

    # Ensure we have seconds_from_snap column
    if 'seconds_from_snap' not in filtered_preds.columns:
        filtered_preds['seconds_from_snap'] = filtered_preds['frames_from_snap'] / 10
    
    # Ensure is_augmented column exists
    if 'is_augmented' not in filtered_preds.columns:
        filtered_preds['is_augmented'] = False
    
    # Round to nearest 0.1 seconds for smoother visualization
    filtered_preds['seconds_rounded'] = np.round(filtered_preds['seconds_from_snap'], 1)
    
    # Group by rounded seconds and calculate mean accuracy
    accuracy_by_frame = filtered_preds.groupby('seconds_rounded')['base_correct'].mean().reset_index()
    
    # Find peaks
    pre_snap_peak = accuracy_by_frame[(accuracy_by_frame['seconds_rounded'] < 0) &
                                      (accuracy_by_frame['seconds_rounded'] >= -8)]['base_correct'].max()

    post_snap_peak = accuracy_by_frame[(accuracy_by_frame['seconds_rounded'] >= 0) & 
                                     (accuracy_by_frame['seconds_rounded'] <= 1.0)]['base_correct'].max()
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    # Plot the line with proper x and y values
    plt.plot(accuracy_by_frame['seconds_rounded'], 
             accuracy_by_frame['base_correct'], 
             marker='o', markersize=4, linestyle='-', color='#1f77b4', label='Accuracy')
    
    # Add vertical line at snap
    plt.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Snap')
    
    # Add horizontal lines for peak accuracies
    plt.axhline(y=pre_snap_peak, color='green', linestyle='-.', linewidth=1.5, alpha=0.7, 
               label=f'Peak Pre-Snap: {pre_snap_peak:.3f}')
    plt.axhline(y=post_snap_peak, color='purple', linestyle='-.', linewidth=1.5, alpha=0.7,
               label=f'Peak Post-Snap: {post_snap_peak:.3f}')
    
    # Add styling
    plt.xlabel('Seconds from Snap', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right')
    
    # Set sensible limits
    plt.xlim(-8, max(accuracy_by_frame['seconds_rounded']))
    plt.ylim(0.65, 1.0)

    
    # Show plot
    plt.tight_layout()
    plt.show()
    
    # Print peak accuracies
    print(f"Peak pre-snap accuracy: {pre_snap_peak:.4f}")
    print(f"Peak post-snap (within 1.0s) accuracy: {post_snap_peak:.4f}")
    
    return accuracy_by_frame

# Example usage
if __name__ == "__main__":
    pred_file = f"C:/python/nfl-big-data-bowl-2025/week_1_preds_mlp.csv"
    df = pd.read_csv(pred_file)
    overall_accuracy = df['pred'] == df['actual']
    overall_accuracy = overall_accuracy.mean()
    print(f"Overall accuracy of test set: {overall_accuracy:.4f}")

    processed_preds, accuracy_data = process_and_visualize(1)
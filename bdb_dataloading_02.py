import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from bdb_cleaning_functions_01 import *

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def process_week_data(week_number, plays):
    # Function to read in all data & apply cleaning functions
    file_path = f"C:/python/nfl-big-data-bowl-2025/tracking_week_{week_number}.csv"
    week = pd.read_csv(file_path)
    print(f"Finished reading Week {week_number} data")

    # applying cleaning functions
    week = rotate_direction_and_orientation(week)
    week = make_plays_left_to_right(week)
    week = calculate_velocity_components(week)
    week = pass_attempt_merging(week, plays)
    week = label_offense_defense_manzone(week, plays)

    week['week'] = week_number
    week['uniqueId'] = week['gameId'].astype(str) + "_" + week['playId'].astype(str)
    week['frameUniqueId'] = (
        week['gameId'].astype(str) + "_" +
        week['playId'].astype(str) + "_" +
        week['frameId'].astype(str))

    # adding frames_from_snap
    snap_frames = week[week['frameType'] == 'SNAP'].groupby('uniqueId')['frameId'].first()
    week = week.merge(snap_frames.rename('snap_frame'), on='uniqueId', how='left')
    week['frames_from_snap'] = week['frameId'] - week['snap_frame']

    # filtering only for even frames
    week = week[week['frameId'] % 2 == 0]

    # ridding of noisier outliers out of scope (15 seconds after the snap)
    week = week[(week['frames_from_snap'] >= -150) & (week['frames_from_snap'] <= 50)]

    # applying data augmentation to increase training size (centered around 0-4 seconds presnap!)
    # -- 1/3rd of the current num of frames... specifically selecting for frames around the snap

    num_unique_frames = len(set(week['frameUniqueId']))
    selected_frames = select_augmented_frames(week, int(num_unique_frames / 3), sigma=5)
    week_aug = data_augmentation(week, selected_frames)

    week = pd.concat([week, week_aug])

    print(f"Finished processing Week {week_number} data")
    print()

    return week

# Read static csv files
games = pd.read_csv("C:/python/nfl-big-data-bowl-2025/games.csv")
player_play = pd.read_csv("C:/python/nfl-big-data-bowl-2025/player_play.csv")
players = pd.read_csv("C:/python/nfl-big-data-bowl-2025/players.csv")
plays = pd.read_csv("C:/python/nfl-big-data-bowl-2025/plays.csv")

all_weeks = []

for week_number in range(1, 2):
    week_data = process_week_data(week_number, plays)
    all_weeks.append(week_data)

all_tracking = pd.concat(all_weeks, ignore_index=True)
all_tracking = all_tracking[(all_tracking['club'] != 'football') & (all_tracking['passAttempt'] == 1)]
     
features = ["x_clean", "y_clean", "v_x", "v_y", "defense"]
target_column = "pff_manZone"

#  Save training, validation, and test set
train_df, test_df, val_df = split_data_by_uniqueId(all_tracking, unique_id_column="frameUniqueId")

cols = ['frameUniqueId', 'displayName', 'frameId', 'frameType', 'x_clean', 'y_clean', 'v_x', 'v_y', 'defensiveTeam', 'pff_manZone', 'defense']

train_df = train_df[cols]
val_df = val_df[cols]
test_df = test_df[cols]

train_features, train_targets = prepare_frame_data(train_df, features, target_column)
val_features, val_targets = prepare_frame_data(val_df, features, target_column)
test_features, test_targets = prepare_frame_data(test_df, features, target_column)

# Save frameUniqueIds for test set (to align with test_features)
# Extract frameUniqueIds in order matching test_features
frame_ids_test = test_df[['frameUniqueId']].drop_duplicates().reset_index(drop=True)
frame_ids_tensor = list(frame_ids_test['frameUniqueId'])

torch.save(frame_ids_tensor, "C:/python/nfl-big-data-bowl-2025/frame_ids_test_preds.pt")

torch.save(train_features, "C:/python/nfl-big-data-bowl-2025/features_training_preds.pt")
torch.save(train_targets, "C:/python/nfl-big-data-bowl-2025/targets_training_preds.pt")

torch.save(val_features, "C:/python/nfl-big-data-bowl-2025/features_val_preds.pt")
torch.save(val_targets, "C:/python/nfl-big-data-bowl-2025/targets_val_preds.pt")

torch.save(test_features, "C:/python/nfl-big-data-bowl-2025/features_test_preds.pt")
torch.save(test_targets, "C:/python/nfl-big-data-bowl-2025/targets_test_preds.pt")

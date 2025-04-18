import pandas as pd

# Check the training annotations file
print("=== Training Annotations File ===")
train_df = pd.read_csv('data/consolidated/how2sign_realigned_train.csv', delimiter='\t', nrows=5)
print("Sample VIDEO_IDs:", train_df['VIDEO_ID'].tolist())
print()

# Check the front landmarks file
print("=== Front Landmarks File ===")
front_df = pd.read_csv('data/consolidated/normalized_landmarks_front.csv', nrows=5)
print("Sample segment_ids:", front_df['segment_id'].tolist())
print()

# Check if the IDs match or need transformation
print("=== ID Analysis ===")
# Extract video ID from segment_id (assuming format: videoID_frameNumber)
front_video_ids = [sid.split('_')[0] if isinstance(sid, str) else '' for sid in front_df['segment_id']]
print("Front landmarks video IDs:", front_video_ids)
print("Do IDs match?", set(front_video_ids).intersection(set(train_df['VIDEO_ID'])))

# Check the side landmarks file
print("=== Side Landmarks File ===")
side_df = pd.read_csv('data/consolidated/normalized_landmarks_side.csv', nrows=1)
print("Columns:", side_df.columns.tolist()[:5], "...") 
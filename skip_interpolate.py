import os
import pandas as pd

# === Step 1: Frame Skipping ===

def skip_frames(input_folder, output_folder, skip_interval=5):
    os.makedirs(output_folder, exist_ok=True)
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace('.txt', '_skipped.txt'))

            with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
                for line in f_in:
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        frame = int(parts[0])
                        if frame % skip_interval == 0:
                            f_out.write(line)

# === Step 2: Interpolation ===

def interpolate_frames(skipped_folder, interpolated_folder, max_frame=None):
    os.makedirs(interpolated_folder, exist_ok=True)
    for file_name in os.listdir(skipped_folder):
        if file_name.endswith('_skipped.txt'):
            input_path = os.path.join(skipped_folder, file_name)
            output_path = os.path.join(interpolated_folder, file_name.replace('_skipped.txt', '_interpolated.txt'))

            # Read data
            df = pd.read_csv(input_path, header=None)
            df.columns = ['frame', 'id', 'x', 'y']

            # Set multi-index
            df.set_index(['id', 'frame'], inplace=True)

            # Reindex: fill in missing frames
            new_index = []
            for obj_id in df.index.get_level_values(0).unique():
                frames = df.loc[obj_id].index
                if max_frame is None:
                    max_f = frames.max()
                else:
                    max_f = max_frame
                new_frames = list(range(0, max_f + 1))
                new_index.extend([(obj_id, f) for f in new_frames])

            df = df.reindex(new_index)

            # Interpolate
            df[['x', 'y']] = df[['x', 'y']].interpolate(method='linear')

            # Drop completely NaN ids (if any)
            df = df.dropna()

            # Save back
            df = df.reset_index()
            df.to_csv(output_path, index=False, header=False)

# === Usage ===

if __name__ == "__main__":
    input_folder = './annotations_with_ids'
    skipped_folder = './output/skipped'
    interpolated_folder = './output/interpolated'

    skip_frames(input_folder, skipped_folder, skip_interval=5)
    interpolate_frames(skipped_folder, interpolated_folder)

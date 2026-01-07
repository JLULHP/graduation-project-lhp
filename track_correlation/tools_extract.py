import os
import pandas as pd

def extract_cooperative_data(source_folder, target_folder, split_type='train'):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    split_folder = os.path.join(source_folder, split_type)
    if not os.path.exists(split_folder):
        print(f"Folder {split_folder} not found, skipping")
        return

    vehicle_base = os.path.join(os.path.dirname(source_folder), "vehicle-trajectories")
    vehicle_folder = os.path.join(vehicle_base, split_type)

    for filename in os.listdir(split_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(split_folder, filename)
            try:
                data = pd.read_csv(file_path)
                extracted_data = data[['timestamp', 'id', 'type', 'x', 'y', 'car_side_id', 'road_side_id']].copy()
                extracted_data['trajectory_id'] = extracted_data.apply(
                    lambda row: f"{int(row['car_side_id'])}_{int(row['road_side_id'])}"
                    if row['road_side_id'] != -1
                    else f"{int(row['car_side_id'])}_car_only",
                    axis=1
                )

                vehicle_file = os.path.join(vehicle_folder, filename)
                if os.path.exists(vehicle_file):
                    vehicle_data = pd.read_csv(vehicle_file)
                    pedestrian_data = vehicle_data[vehicle_data['type'].isin(['PEDESTRIAN', 'BICYCLE', 'CYCLIST'])].copy()

                    if not pedestrian_data.empty:
                        pedestrian_data['trajectory_id'] = pedestrian_data['id']
                        pedestrian_data['car_side_id'] = pedestrian_data['id']
                        pedestrian_data['road_side_id'] = -1

                        pedestrian_extracted = pedestrian_data[['timestamp', 'id', 'type', 'x', 'y', 'car_side_id', 'road_side_id', 'trajectory_id']].copy()
                        extracted_data = pd.concat([extracted_data, pedestrian_extracted], ignore_index=True)

                cols = ['trajectory_id', 'timestamp', 'id', 'type', 'x', 'y', 'car_side_id', 'road_side_id']
                extracted_data = extracted_data[cols]
                extracted_data = extracted_data.sort_values(['trajectory_id', 'timestamp'])

                new_filename = f"{os.path.splitext(filename)[0]}_{split_type}.csv"
                new_file_path = os.path.join(target_folder, new_filename)
                extracted_data.to_csv(new_file_path, index=False)
                print(f"Processed {filename} -> {len(extracted_data)} points")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

    print(f"Completed processing {split_type} dataset")

if __name__ == "__main__":
    base_path = r"D:\DAIR-V2X数据集\车路协同轨迹预测v2x-Seq-TFD\V2X-Seq-TFD\cooperative-vehicle-infrastructure"
    source_folder = os.path.join(base_path, "cooperative-trajectories")
    processed_base = r"D:\DAIR-V2X数据集\车路协同轨迹预测v2x-Seq-TFD\V2X-Seq-TFD\processed_data"

    train_target = os.path.join(processed_base, "train")
    extract_cooperative_data(source_folder, train_target, 'train')

    val_target = os.path.join(processed_base, "val")
    extract_cooperative_data(source_folder, val_target, 'val')

    print("All data extraction completed")

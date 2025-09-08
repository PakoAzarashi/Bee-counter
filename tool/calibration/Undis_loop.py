import os

video_files = ["F:/final_correlation_data"]

for video_file in video_files:
    print(video_file)
    os.system(f"python video_Undis.py --Input_dir {video_file}")
    
import sys
import pandas as pd
from pytubefix import YouTube
from moviepy import VideoFileClip


def generate(original_data, new_data, dataset_folder):
    original_data = pd.read_csv(original_data)
    new_data = pd.read_csv(new_data)

    generated = []
    for index, row in new_data.iterrows():
        # data related to img and vid paths
        actual_index = "0" + str(20 + index % 5)
        imgPath = (
            "/images/"
            + row["action"]
            + "/"
            + row["action"]
            + "_"
            + actual_index
            + ".png"
        )
        vidPath = (
            "/video/"
            + row["action"]
            + "/"
            + row["action"]
            + "_"
            + actual_index
            + ".mp4"
        )

        # get video
        video_url = row["link"]
        yt = YouTube(video_url)
        ys = yt.streams.get_highest_resolution()

        # download video
        desired_filename = row["action"] + "_" + actual_index + ".mp4"
        download_directory = dataset_folder + "/video/" + row["action"] + "/"
        full_path = download_directory + desired_filename
        temp_directory = dataset_folder + "/temp/"
        ys.download(output_path=temp_directory, filename=desired_filename)

        # trim video
        video = VideoFileClip(temp_directory + desired_filename)
        start = sum(
            int(t) * x
            for t, x in zip(
                row["start_time"].split(":")[::-1],
                [1, 60, 3600][: len(row["start_time"].split(":"))],
            )
        )
        stop = sum(
            int(t) * x
            for t, x in zip(
                row["end_time"].split(":")[::-1],
                [1, 60, 3600][: len(row["end_time"].split(":"))],
            )
        )

        trimmed_video = video.subclipped(start, stop)
        trimmed_video.write_videofile(full_path)

        # create new data sample
        data = {
            "action": row["action"],
            "sample": actual_index,
            "imgPath": imgPath,
            "vidPath": vidPath,
            "dataset": row["dataset"],
            "one_person": row["one_person"],
            "face_visible": row["face_visible"],
            "gender": row["gender"],
            "age": row["age"],
            "race": row["race"],
        }

        # add to list of generated samples
        generated.append(data)

    # save new csv with all data
    final = pd.DataFrame(generated)
    final_combined = pd.concat([original_data, final], ignore_index=True)
    final_combined.to_csv(index=False, path_or_buf="combined.csv")


def main():
    # read the original data and saves to the new data
    original_data = sys.argv[1]
    new_data = sys.argv[2]
    dataset_folder = sys.argv[3]

    # run generation
    generate(original_data, new_data, dataset_folder)


if __name__ == "__main__":
    main()

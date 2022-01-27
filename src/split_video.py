import os
import subprocess
import sys
import numpy as np
import cv2
from pydub import AudioSegment
import moviepy.editor as mp

# ------ MACROS ------
CV2_FRAME_RATE =  cv2.CAP_PROP_FPS
SIM_FRAME_RATE = 30


def main():

    # use -rmi flag to remove images
    if '-rmi' in sys.argv:
        remove_all_imgs()

    # convert all videos in /video to mp4
    if '-c' in sys.argv:
        convert_to_mp4()

    # controls frame rate
    if '-fps' in sys.argv:
        SIM_FRAME_RATE = sys.argv[sys.argv.index('-fps') + 1]

    if '-sv' in sys.argv:
        split_video()

    # remove dead frames
    if '-rdf' in sys.argv:
        cam = cv2.VideoCapture("../data/video/walking.mp4")
        video_info = get_duration(cam)
        remove_dead_frames(video_info)

    #split_video()
    #remove_dead_frames()


def convert_to_mp4():
    """
    Converts all videos in ../data/video to .mp4

    Flags: -rm removes original video
    """
    src = '../data/video'
    dst = '../data/video'

    for root, dirs, filenames in os.walk(src, topdown=False):
        print(src)
        for filename in filenames:
            print('[INFO] 1', filename)
            try:
                _format = ''
                if ".flv" in filename.lower():
                    _format = ".flv"
                if ".mp4" in filename.lower():
                    _format = ".mp4"
                if ".avi" in filename.lower():
                    _format = ".avi"
                if ".mov" in filename.lower():
                    _format = ".mov"

                inputfile = os.path.join(root, filename)
                outputfile = os.path.join(
                    dst, filename.lower().replace(_format, ".mp4"))

                # dependent on ffmpeg package
                subprocess.call(['ffmpeg', '-i', inputfile, outputfile])

                # removes original video
                ext = outputfile.split('.')[-1]
                if os.path.exists(outputfile) and ext != 'mp4' and '-rm' in sys.argv:
                    subprocess.call(['rm', inputfile])
            except Exception as e:
                print("An exception occurred: {}".format(e))


def split_video():
    """
    Split video in ../data/video into frames saved to ../data/imgs
    """

    # Read the video from specified path
    input_dir = '../data/video/'
    output_dir = '../data/imgs/'

    cam = cv2.VideoCapture(input_dir + "walking.mp4")
    print("duration\n")

    video_info = get_duration(cam)

    # worked = cam.set(FRAME_RATE, 1)
    # print("Worked: {}\n\n\n".format(worked))

    # frame
    currentframe = 0

    # simulates frame rate
    frame_rate = 10
    prev = 0

    while True:
        # control frame rate for *live feed*
        # time_elapsed = time.time() - prev

        # reading from frame
        ret, frame = cam.read()

        if not ret:
            break

        # control frame rate for live feed
        # if ret and time_elapsed > 1./frame_rate:
        # prev = time.time()

        # if video is still left continue creating images
        # total_frames = get_duration(cam)['frame_count']
        # duration_s = get_duration(cam)['duration_s']
        # frame_selector = total_frames / (duration_s * SIM_FRAME_RATE)


        # failed attempt to simulate frame rate
        # if currentframe % frame_selector == 0:
        name = output_dir + 'frame-' + str(currentframe) + '.jpg'
        print ('Creating...' + name)

        # writing the extracted images
        cv2.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def get_duration(cam):
    """
    Retrieve basic information about video
    """

    fps = cam.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration_s': duration,
        'seconds': duration % 60,
        'minutes': int(duration / 60)
    }


def remove_all_imgs():
    """
    Remove all images from ../data/imgs
    """

    dir_name = "../data/imgs"
    test = os.listdir(dir_name)

    for item in test:
        if item.endswith(".jpg"):
            os.remove(os.path.join(dir_name, item))


def remove_dead_frames(video_info):
    dir_name = "../data/imgs/"
    image_directory_content = os.listdir(dir_name)
    frames = list(filter(lambda image: image.endswith(".jpg"), image_directory_content))
    frames.sort()
    frame_increment = int(video_info.get("fps")/2)
    percent_diff_threshold = 3

    active_clips = []

    active_audio = []

    recording_audio = False

    print("Analyzing dead frames")

    for frame_number in range(0, len(frames), frame_increment):
        if frame_number + frame_increment < len(frames):
            img1 = cv2.imread(dir_name + "frame-" + str(frame_number) + ".jpg")
            img2 = cv2.imread(dir_name + "frame-" + str(frame_number + frame_increment) + ".jpg")
            absolute_difference = cv2.absdiff(img1, img2).astype(np.uint8)
            threshold_difference = cv2.threshold(absolute_difference, 10, 255, cv2.THRESH_BINARY)[1]
            percent_diff = (np.count_nonzero(threshold_difference) * 100) / absolute_difference.size
            if percent_diff > percent_diff_threshold:
                if not recording_audio:
                    timestamp = (frame_number / video_info.get("fps"))
                    recording_audio = True
                    active_audio.append(timestamp * 1000)
                    print(timestamp)

                for frame in range(frame_number, frame_number + frame_increment):
                    img = cv2.imread(dir_name + "frame-" + str(frame) + ".jpg")
                    active_clips.append(img)
            else:
                if recording_audio:
                    timestamp = (frame_number / video_info.get("fps"))
                    recording_audio = False
                    active_audio.append(timestamp * 1000)
                    print(timestamp)

    height, width, layers = active_clips[0].shape
    size = (width, height)
    video_writer = cv2.VideoWriter("../data/output/vid_no_audio.mp4", cv2.VideoWriter_fourcc(*'mp4v'), video_info.get("fps"), size)

    print("Creating cleaned video")
    for frame in active_clips:
        video_writer.write(frame)

    # EXTRACTING AUDIO
    movie_py_video = mp.VideoFileClip(r"../data/video/walking.mp4")
    movie_py_video.audio.write_audiofile(r"../data/output/my_audio.mp3")
    all_audio = AudioSegment.from_mp3("../data/output/my_audio.mp3")
    for time in range(0, len(active_audio), 2):
        live_audio = all_audio[active_audio[time]: active_audio[time + 1]]
        live_audio.export("../data/output/trimmed_audio.mp3", format="mp3")

    os.remove("../data/output/my_audio.mp3")

    video_writer.release()

    subprocess.call(["ffmpeg",
                     "-i",  "../data/output/vid_no_audio.mp4",
                     "-i", "../data/output/trimmed_audio.mp3",
                     "-c:v", "copy",
                     "-map", "0:v",
                     "-map", "1:a",
                     "-shortest", "../data/output/final.mp4"])




main()

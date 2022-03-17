import boto3
import cv2
import model
import os

session = boto3.Session(profile_name='stvrain')
s3_client = session.client('s3')
s3 = session.resource('s3')

SOURCE_VIDEO_BUCKET = 'dxhub-svvsd-video-mp4'
OUTPUT_IMAGE_BUCKET = 'dxhub-svvsd-unlabeled-images'

TEMP_DIRECTORY = './tmp/'


def detect_fight(file_name):
    try:
        # print("Downloading " + file_name)
        # s3_client.download_file(Bucket=SOURCE_VIDEO_BUCKET,
        #                         Key=file_name,
        #                         Filename=TEMP_DIRECTORY + file_name)
        print("Checking for fights...")
        fight_detected = check_frames(file_name)

        if fight_detected:
            print("Fight Detected")

        else:
            print("Fight Not Detected")

    except Exception as e:
        print("An exception occurred: {}".format(e))


def check_frames(file_name):
    """
    Split video in ../data/video into frames saved to ../data/imgs
    """

    # Read the video from specified path
    cam = cv2.VideoCapture(TEMP_DIRECTORY + file_name)

    current_frame = 0

    target_frame = 0

    frame_increment = 5

    num_frame = 0
    detection_window = [0, 0, 0, 0, 0]

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        if current_frame == target_frame:
            frame_name = file_name.replace(".mp4", "") + '_frame_' + str(current_frame) + '.jpg'
            print("Analyzing " + frame_name + "...")
            cv2.imwrite(TEMP_DIRECTORY + frame_name, frame)
            target_frame += frame_increment

            fight_detected = model.run_model(TEMP_DIRECTORY + frame_name)

            window_index = num_frame % 5
            if fight_detected:
                detection_window[window_index] = 1
            else:
                detection_window[window_index] = 0

            if sum(detection_window) == 3:
                return True

            num_frame += 1

        current_frame += 1

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

    return False


def main():
    detect_fight("ehs wo#230844 cr#182333.mp4")



main()

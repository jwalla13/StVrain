# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-custom-labels-developer-guide/blob/master/LICENSE-SAMPLECODE.)

import time
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
import io
import boto3
import cv2

session = boto3.Session(profile_name='stvrain')
model_client = session.client('rekognition')

MODEL_ARN = "arn:aws:rekognition:us-west-2:099295524168:project/st-vrain-fight-detection-4/version/st-vrain-fight-detection-4.2022-03-03T14.40.28/1646347228566"
MIN_CONFIDENCE = 85

def start_model():
    project_arn = 'arn:aws:rekognition:us-west-2:099295524168:project/st-vrain-fight-detection-4/1646344074952'
    min_inference_units = 1
    version_name='st-vrain-fight-detection-4.2022-03-03T14.40.28'

    try:
        # Start the model
        print('Starting model: ' + MODEL_ARN)
        response = model_client.start_project_version(
            ProjectVersionArn=MODEL_ARN, MinInferenceUnits=min_inference_units)
        # Wait for the model to be in the running state
        project_version_running_waiter = model_client.get_waiter(
            'project_version_running')
        project_version_running_waiter.wait(
            ProjectArn=project_arn, VersionNames=[version_name])

        # Get the running status
        describe_response = model_client.describe_project_versions(ProjectArn=project_arn,
                                                                   VersionNames=[version_name])
        for model in describe_response['ProjectVersionDescriptions']:
            print("Status: " + model['Status'])
            print("Message: " + model['StatusMessage'])
    except Exception as e:
        print(e)

    print('Done...')


# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# PDX-License-Identifier: MIT-0 (For details, see https://github.com/awsdocs/amazon-rekognition-custom-labels-developer-guide/blob/master/LICENSE-SAMPLECODE.)


def save_image(photo, response, bucket=None):
    if bucket:
        # Load image from S3 bucket
        s3_connection = boto3.resource('s3')

        s3_object = s3_connection.Object(bucket, photo)
        s3_response = s3_object.get()

        stream = io.BytesIO(s3_response['Body'].read())
    else:
        image = open(photo, 'rb')
        stream = io.BytesIO(image.read())

    image = Image.open(stream)

    # Ready image to draw bounding boxes on it.
    imgWidth, imgHeight = image.size
    draw = ImageDraw.Draw(image)

    # calculate and display bounding boxes for each detected custom label
    print('Detected custom labels for ' + photo)
    for customLabel in response['CustomLabels']:
        # print('Label ' + str(customLabel['Name']))
        # print('Confidence ' + str(customLabel['Confidence']))
        if 'Geometry' in customLabel:
            box = customLabel['Geometry']['BoundingBox']
            left = imgWidth * box['Left']
            top = imgHeight * box['Top']
            width = imgWidth * box['Width']
            height = imgHeight * box['Height']

            fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 50)
            draw.text(
                (left, top), customLabel['Name'], fill='#00d400', font=fnt)

            # print('Left: ' + '{0:.0f}'.format(left))
            # print('Top: ' + '{0:.0f}'.format(top))
            # print('Label Width: ' + "{0:.0f}".format(width))
            # print('Label Height: ' + "{0:.0f}".format(height))

            points = (
                (left, top),
                (left + width, top),
                (left + width, top + height),
                (left, top + height),
                (left, top))
            draw.line(points, fill='#00d400', width=5)

    # Save image with boxes drawn
    image.show()
    image.save(photo)



def run_model(photo, bucket=None):
    # Call DetectCustomLabels
    if bucket is not None:
        response = model_client.detect_custom_labels(Image={'S3Object': {'Bucket': bucket, 'Name': photo}},
                                                     MinConfidence=MIN_CONFIDENCE,
                                                     ProjectVersionArn=MODEL_ARN)

    # For object detection use case, uncomment below code to display image.
    else:
        image = open(photo, 'rb')
        image_bytes = image.read()
        response = model_client.detect_custom_labels(Image={"Bytes": image_bytes},
                                                     MinConfidence=MIN_CONFIDENCE,
                                                     ProjectVersionArn=MODEL_ARN)

    if len(response['CustomLabels']):
        print("Found Fight in Frame: " + photo)
        save_image(photo, response)

    return len(response['CustomLabels'])


def stop_model():
    print('Stopping model:' + MODEL_ARN)

    # Stop the model
    try:
        response = model_client.stop_project_version(ProjectVersionArn=MODEL_ARN)
        status = response['Status']
        print('Status: ' + status)
    except Exception as e:
        print(e)

    print('Done...')




import os
import pandas as pd
import cv2
import json
import requests
import csv
import time
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.face import FaceClient
from azure.ai.vision.face.models import (
    FaceDetectionModel,
    FaceRecognitionModel,
    FaceAttributeTypeDetection03,
    FaceAttributeTypeRecognition04,
)
import joblib

# Set environment variable for QT platform
os.environ['QT_QPA_PLATFORM'] = 'xcb'

def categorize_head_pose(pitch, yaw):
    if abs(yaw) < 10 and abs(pitch) < 10:
        return 'forward'
    elif pitch < -10:
        return 'down'
    elif yaw > 10:
        return 'right'
    elif yaw < -10:
        return 'left'
    else:
        return 'unknown'

# Load credentials
credential = json.load(open("credential.json"))
key = credential["FACE_API_KEY"]
endpoint = credential["ENDPOINT_FACE"]

# Create directories for saving frames and metadata
if not os.path.exists('frames'):
    os.makedirs('frames')

# Metadata CSV setup
metadata_file_exists = os.path.isfile('frame_metadata.csv')
metadata_file = open('frame_metadata.csv', mode='a', newline='')
metadata_writer = csv.writer(metadata_file)
if not metadata_file_exists:
    metadata_writer.writerow(['filename', 'timestamp', 'attention'])

# CSV file setup for features
feature_file_exists = os.path.isfile('attention_data2.csv')
feature_csv_file = open('attention_data2.csv', mode='a', newline='')
csv_writer = csv.writer(feature_csv_file)
if not feature_file_exists:
    csv_writer.writerow([
        'filename', 'timestamp', 'attention',
        'face_x', 'face_y', 'face_w', 'face_h',
        'pose_x', 'pose_y', 'pose_down', 'pose_forward', 'pose_left', 'pose_right',
        'pupilLeft_x', 'pupilLeft_y', 'pupilRight_x', 'pupilRight_y',
        'eyeLeftOuter_x', 'eyeLeftOuter_y', 'eyeLeftInner_x', 'eyeLeftInner_y',
        'eyeRightOuter_x', 'eyeRightOuter_y', 'eyeRightInner_x', 'eyeRightInner_y'
    ])

# Initialize FaceClient
face_client = FaceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def process_image(image_path, attention_label):
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image {image_path}")
        return

    # Generate a unique timestamp
    timestamp = time.time()

    # Read image file
    with open(image_path, "rb") as fd:
        file_content = fd.read()

    # Detect faces
    result = face_client.detect(
        file_content,
        detection_model=FaceDetectionModel.DETECTION_03,
        recognition_model=FaceRecognitionModel.RECOGNITION_04,
        return_face_id=False,
        return_face_attributes=[
            FaceAttributeTypeDetection03.HEAD_POSE,
            FaceAttributeTypeRecognition04.QUALITY_FOR_RECOGNITION,
        ],
        return_face_landmarks=True,
        return_recognition_model=True,
        face_id_time_to_live=120,
    )

    features = {}

    for idx, face in enumerate(result):
        face_rectangle = face.face_rectangle
        top = face_rectangle.top
        left = face_rectangle.left
        width = face_rectangle.width
        height = face_rectangle.height

        features["face_x"] = left
        features["face_y"] = top
        features["face_w"] = width
        features["face_h"] = height

        if face.face_attributes:
            head_pose = face.face_attributes.head_pose

            features["pose"] = categorize_head_pose(head_pose.pitch, head_pose.yaw)
            features["pose_x"] = head_pose.yaw
            features["pose_y"] = head_pose.pitch

            # One-hot encode the 'pose' column
            pose_category = categorize_head_pose(head_pose.pitch, head_pose.yaw)
            pose_columns = ['pose_down', 'pose_forward', 'pose_left', 'pose_right']
            for col in pose_columns:
                features[col] = 1 if f'pose_{pose_category}' == col else 0

        if face.face_landmarks:
            landmarks = face.face_landmarks.as_dict()
            features["pupilLeft_x"] = landmarks["pupilLeft"]["x"]
            features["pupilLeft_y"] = landmarks["pupilLeft"]["y"]
            features["pupilRight_x"] = landmarks["pupilRight"]["x"]
            features["pupilRight_y"] = landmarks["pupilRight"]["y"]
            features["eyeLeftOuter_x"] = landmarks["eyeLeftOuter"]["x"]
            features["eyeLeftOuter_y"] = landmarks["eyeLeftOuter"]["y"]
            features["eyeLeftInner_x"] = landmarks["eyeLeftInner"]["x"]
            features["eyeLeftInner_y"] = landmarks["eyeLeftInner"]["y"]
            features["eyeRightOuter_x"] = landmarks["eyeRightOuter"]["x"]
            features["eyeRightOuter_y"] = landmarks["eyeRightOuter"]["y"]
            features["eyeRightInner_x"] = landmarks["eyeRightInner"]["x"]
            features["eyeRightInner_y"] = landmarks["eyeRightInner"]["y"]

            # Write to CSV
            csv_writer.writerow([
                image_path, timestamp, attention_label,
                features["face_x"], features["face_y"], features["face_w"], features["face_h"],
                features["pose_x"], features["pose_y"],
                features['pose_down'], features['pose_forward'], features['pose_left'], features['pose_right'],
                features["pupilLeft_x"], features["pupilLeft_y"], features["pupilRight_x"], features["pupilRight_y"],
                features["eyeLeftOuter_x"], features["eyeLeftOuter_y"], features["eyeLeftInner_x"], features["eyeLeftInner_y"],
                features["eyeRightOuter_x"], features["eyeRightOuter_y"], features["eyeRightInner_x"], features["eyeRightInner_y"]
            ])

            # Write metadata
            metadata_writer.writerow([image_path, timestamp, attention_label])

async def main():
    image_folder = 'not_attention_frames_person2'  # Replace with your folder path containing images
    attention_label = 0  # All images in the folder are labeled as attention

    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
            process_image(image_path, attention_label)

    feature_csv_file.close()
    metadata_file.close()

# Run the main function
asyncio.run(main())

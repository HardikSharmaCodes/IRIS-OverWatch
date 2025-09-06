# IRIS-OverWatch
IRIS OVER-WATCH

IRIS OVER-WATCH is an advanced traffic monitoring system designed to detect, track, and recognize vehicles in real-time. It uses YOLO for accurate object detection, EasyOCR for reading license plates, and DeepSORT for tracking vehicles across video frames. The system assigns a unique ID to each vehicle and keeps a record of all license plates observed, even after the vehicle has left the scene.

The detected vehicles and their corresponding license plate information are displayed on the video with bounding boxes and unique IDs, while the most common license plate prediction for each vehicle is stored in a SQLite database. This enables easy retrieval and analysis of traffic data for research or monitoring purposes.

Key Features:

Real-time detection of cars, trucks, buses, and motorbikes

Unique tracking of each vehicle with persistent IDs

Automatic license plate recognition and memory

Storage of vehicle ID and license plate data in a database

Visualization of bounding boxes, IDs, and license plates on video

Usage Instructions:

Clone the repository to your local machine.

Install required packages using: pip install -r requirements.txt

Download the required YOLO models and place them in the appropriate folder.

Run python traffic_system.py to start the system and process traffic videos.

Note: This project is intended for educational and research purposes and demonstrates an end-to-end solution for traffic monitoring and vehicle identification.

# Fatigue-Detection
Eye state classification using OpenCV and DLib to estimate Percentage Eye Closure (PERCLOS) and alert a drowsy person (such as a driver).

## Face and Eye detector + Alarm.py
Uses OpenCV Haar Cascade classifiers to detect face, and then for eyes within a rough area defined inside the face bounding-box. Eye state classification is done by thresholding the image and counting the number of black pixels (open eye has more black pixels due to the iris)

## source.cpp
Uses DLib facial landmark detector to find the major and minor axes of eyes, as well as mouth. The aspect ratio of major to minor axes is used to determine whether eye/mouth is open; which allows for eye-state classification and yawning detection.
Requires a pre-trained DLib facial landmark detector model in a .dat file. 

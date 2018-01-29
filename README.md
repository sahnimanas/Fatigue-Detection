# Fatigue-Detection
Eye state classification using OpenCV and DLib to estimate Percentage Eye Closure (PERCLOS) and alert a drowsy person (such as a driver).

##### Dependencies:
... 1. OpenCV (3.0 or later)
... 2. Dlib (19.0 or later, for facial landmarking)

#### Using aspect ratio of major/minor axes (cpp)
Uses DLib facial landmark detector to find the major and minor axes of eyes, as well as mouth. The aspect ratio of major to minor axes is used to determine whether eye/mouth is open; which allows for eye-state classification and yawning detection.
Requires a pre-trained DLib facial landmark detector model in a .dat file. 

![alt text][landmarks]

[landmarks]: https://raw.githubusercontent.com/sahnimanas/Fatigue-Detection/master/landmarks.png "Facial landmarks"

#### Using binary thresholding
Uses OpenCV Haar Cascade classifiers to detect face, and then for eyes within a rough area defined inside the face bounding-box. Eye state classification is done by thresholding the image on skin color and counting the number of black pixels, with the threshold normalized for skin color via HSV histogram

![alt text][landmarks]

[landmarks]: https://raw.githubusercontent.com/sahnimanas/Fatigue-Detection/master/binary.png "Binary thresholding"

# ear_matching
Code for CS640 project - ear image matching
Dharmit Dalvi and James Dunn
March/April 2019
Boston University

EXTERNAL DEPENDENCIES. These are REQUIRED for running the code. Make sure to download them!
numpy
cv2
glob
os
matplotlib
pandas
sklearn
copy
Testing was done using Python 3.6.5 using the Spyder IDE on Windows 10 and iOS

RUNNING THE CODE
1. Imagery must be copied into the data folder.
2. Run main.py from your python environment

CHANGING PARAMETERS
parameters.py contans all of the parameters that can be adjusted by the user. The default values work well for images "in the wild". If running with images with the donut device, set DONUT to True. For best results with the donut device, set USE_KEYPOINT_FILE to False, set REMOVE_BACKGROUND to False, and set EDGE_DILATION_DIAMETER to 0.01. You may change other parameters at your discretion.

DESCRIPTION
This library reads in photos of human ears and attempts to match them.  The input data is two sets of images: 1. pictures of ears taken on one day, 2. pictures of the same ears taken on another day

The code reads all the images, performs preprocessing, runs a comparison algorithm to get a "similarity score" or correlation value ranging from 0 (no match) to 1 (perfect match).

Each ear photo from the first day is compared to each ear photo from the second day, and a similiarity score is calculated.  If the photo of the same ear on the second day has the highest similarity score of all the second day photos, then the algorithm has found the correct match. 

"Accuracy" is the number of correct matches made divided by the total number of ears in the dataset.

The "average rank of true match" metric is a measure of how close the algortihm is to getting the correct matches.  1 is perfect, higher than 1 is worse.  Specifically, the "rank of true match" for an ear from the first day is simply how many ears got a higher similarity score than the correct ear from the second day, plus 1.
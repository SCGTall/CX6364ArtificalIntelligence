For submission of team project, I include my faces and my friends' faces. THESE IMAGES SHOULD NOT BE BROADCASTED IN PUBLIC OR USED IN COMMERCIAL USAGE!!!

Environment:

To settle down the environment, it will be a slightly different:

python 3.6 for venv (use anaconda)
# opencv
pip3 install opencv-python
# dlib
brew install cmake
brew install boost
#brew install boost-python
brew install dlib
pip3 install numpy
pip3 install scipy
pip3 install scikit-image
pip3 install dlib  # wait long for this

BTW, you need to install other necessary libraries like os, numpy. These are common and can be automatically solved by IDE.


Running:
You are suggested to use your own faces to have more fun. You can follow below steps to explore our project:

Step 1: prepare data (You can skip this if you want to directly use the given data)
1) run get_my_faces.py to capture faces from laptop camera. You can adjust start_num, end_num and output_dir to flexibly get faces for training and testing.
2) download images from open source library. For me, I use lfw.tgz from UMass.
3) run set_other_faces.py to capture faces from photos.

Step 2: training data
1) run train_model.py to train the model. You can find most useful parameters in faces_model.py

Step 3: testing data (This two scripts will load pre-trained model saved by train_model.py)
1) run is_my_face_camera_ver.py load pre-trained model and predict the faces capture in camera. You can adjust limit for dataset size.
2) run is_my_face_load_ver.py load pre-trained model and predict the testing set.

12/12/2021 17:02:35 By Chaoran Li

For submission of team project, I include my faces and my friends' faces. THESE IMAGES SHOULD NOT BE BROADCASTED IN PUBLIC OR USED IN COMMERCIAL USAGE!!!

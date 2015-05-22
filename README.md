# vocabBuild

## Compile Instructions
LDFLAGS='-lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_core -lopencv_nonfree -fopenmp -lboost_system -lboost_timer'

g++-4.9 indexFlann.cpp $LDFLAGS -o indexFlann -std=c++11

## Running Instructions
./indexFlann -i ~/deb/distractorSURFFeatures.yml -q ~/deb/query100SURFDescriptors.yml -o ~/deb/parallelIndexFlann.data 4 10 3072


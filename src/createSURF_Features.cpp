/** //
//  createSIFT_Features.cpp
//  
//
//  Created by Deb on 6/4/15.
 Arguments for this code:
    *   -i [/full/path/to/filelist.txt]
    *   -o [/full/path/to/SURFdescriptors.yml]
    *   -A [Optional Argument to choose feature descriptor ["SURF","SIFT","AKAZE"] #NOT YET IMPLEMENTED
    - This program reads in an input file listing images in .jpg format.
    - Each of these images are processed to compute the SIFT Feature detectors
    - The SIFT features are stored into a csv file containing the feature descriptors
 

**/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <vector>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void readme();
std::vector<std::string> readfilelist(std::string infile);


bool isLandscape(const cv::Mat &image) {
    return image.cols > image.rows;
}

//Resize Function
cv::Mat &resizeToFit(cv::Mat &image, CvSize size) {
    //if it is a smaller image, no resize is needed
    if (image.rows < size.height && image.cols < size.width) {
        return image;
    }
    else {
        //determine which side to fit the scale
        double scale_ratio = 1.0;
        if (image.cols > image.rows)
            scale_ratio = (double)size.width / (double)image.cols;
        else {
            scale_ratio = (double)size.height / (double)image.rows;
        }
        cv::resize(image, image, cvSize(image.cols * scale_ratio, image.rows * scale_ratio));
    }
    return image;
}

//Write Image to File
void writeMatToFile(cv::Mat& m, const string* filename)
{
    ofstream fout(*filename);
    
    if (!fout) {
        cout << "File Not Opened" << endl;
        return;
    }
    
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            fout << m.at<float>(i, j) << "\t";
        }
        fout << endl;
    }
    
    fout.close();
}

void writeMatFileStorage(cv::Mat& m, const string* filename)
{
    FileStorage fs(*filename, FileStorage::WRITE);
    fs<<"Descriptors"<<m;
}

//Get Image Id
unsigned int get_imageid(const string fname){
    //fname.strmatch();
    int curr_id;
    unsigned foundslash = fname.find_last_of("/");
    //unsigned founddot = fname.find_last_of(".");
    string curr_fname = fname.substr(foundslash+1);
    curr_fname.erase(curr_fname.end()-3,curr_fname.end());
    const char* cfname = curr_fname.c_str();
    curr_id=atoi(cfname);
//    cout<<"Returned id: "<<curr_id<<endl;
    return curr_id;
    
    //return 0;
}

int main(int argc, char** argv)
{
    if( argc < 3 )
    { readme(); return -1; }
    
    std::cout<<"Instantiating Variables..."<<std::endl;
    //Instantiate Variables
    std::string infile = argv[2];
    std::string outfile = argv[4];
    
    std::cout<<"Reading input image list from :"<<infile<<std::endl;
    std::cout<<"Descriptors will be stored in: "<<outfile<<std::endl;

    int minHessian = 300;
    SurfFeatureDetector detector(minHessian, 4, 2 ,true);
    SurfDescriptorExtractor extractor;
    
    Mat img, img_keypoints;
    Mat descriptor;
    
    std::vector<KeyPoint> keypoints_1;
    //std::tuple<std::vector<KeyPoint>, int>;
    
    //FeatureDB surfdesciptors;
    Mat surfdescriptors;
    
  
    
    // Read the list of files from input file
    std::vector<std::string> Filelist = readfilelist(infile);
    std::cout<<"Number of images to process: "<<Filelist.size()<<std::endl;
    // Compute the SURF Descriptors for each image in the filelist
    for (int i=0; i<Filelist.size();i++) {
        //std::cout<<"Processing Image: "<<Filelist[i]<<std::endl;
        unsigned int img_id = get_imageid(Filelist[i]);
        img = imread(Filelist[i], CV_LOAD_IMAGE_GRAYSCALE);
        if(isLandscape(img))
            img=resizeToFit(img, cvSize(640,480));
        else
            img=resizeToFit(img, cvSize(480, 640));
        
        detector.detect(img, keypoints_1); //Get keypoints

        //Compute Descriptors
        extractor.compute(img, keypoints_1, descriptor);
        
        //Store the descriptor
        //surfdesciptors.push_back(descriptor, img_id);
        Mat cols(descriptor.rows, 1,CV_32F, img_id); //Create a column of image ids
//         printf("Data type for cols : %d \n",cols.type());
        //cols.setTo(img_id);

        cv::hconcat(descriptor, cols, descriptor);
         if (i==0) {
            surfdescriptors=descriptor;
        }
        else
            surfdescriptors.push_back(descriptor);
        
        //std::cout<<keypoints_1<<endl;
        
        //drawKeypoints(img, keypoints_1, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
        //imshow("Image Keypoints", img_keypoints);
        //waitKey();
    }
    std::cout<<"Finished Processing Images..."<<std::endl;
    
    //writeMatToFile(surfdescriptors, &outfile); //Write descriptors to file
    std::cout<<"Writing feature to file : "<<outfile<<std::endl;
    writeMatFileStorage(surfdescriptors, &outfile);
    std::cout<<"Done. Exiting Program"<<std::endl;

    return 0;

}

std::vector<std::string> readfilelist(std::string infile) //Function to read files from list into vector<string>
{
    std::ifstream input;
    std::vector<std::string> FileList;
    
    std::string curr_file;
    input.exceptions ( std::ifstream::failbit | std::ifstream::badbit );
    try {
        input.open(infile,std::ios::in);
        
        while (!input.eof()) {
            getline(input, curr_file);
            FileList.push_back(curr_file);
            //std::cout<<"Parsed image locations"<<std::endl;
        }
    }
    catch(std::ifstream::failure e){
        std::cerr<<"Error Parsing input file"<<std::endl;
    }
    input.close();
    return FileList;
    
}

void readme() //Function to display help
{
    std::cout<<" Usage: ./createSIFT_Features -i </full/path/to/image/filelist.txt> -o </full/path/to/SIFTdescriptors.yml> "<<std::endl;
}
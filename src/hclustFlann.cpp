//
//  indexFlann.cpp
//  SIFT_FeatureGenerator
//
//  Created by Debaditya on 8/4/15.
//  Copyright (c) 2015 Debaditya. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <vector>
#include <fstream>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <flann/flann.hpp>
#include <boost/timer/timer.hpp>
#include <cmath>

#include "map_Utils.h"
#include "minicsv.h"
using namespace std;
//using namespace cv;

void readme() //Function to display help
{
    std::cout<<" Usage: ./indexFlann -i </full/path/to/inputfeatures.yml> -o </full/path/to/index.data> clusters branches iterations"<<std::endl;
}

 int main(int argc, char** argv)
{
    if( argc < 3 )
    { readme(); return -1; }
    
    std::cout<<"Instantiating Variables..."<<std::endl;
    //Instantiate Variables
    std::string infile = argv[2];
    std::string outfile = argv[4];
    int clusters=atoi(argv[5]);
    int branches = atoi(argv[6]);
    int iterations = atoi(argv[7]);
    
    std::cout<<"Reading input image descriptors from :"<<infile<<std::endl;
    std::cout<<"Index will be stored in: "<<outfile<<std::endl;
    
    
    //Load the input descriptors:
    cv::Mat descriptors_orig=loadDescriptors(&infile);
    cv::Mat descriptors;
        descriptors=descriptors_orig.colRange(0,descriptors_orig.cols-1); //To get rid of the last column
        descriptors=descriptors.clone();
        //descriptors=descriptors_orig.clone();
    printf("Input features: %i X %i\n", descriptors.rows, descriptors.cols);
    
    //Load the additional 1000 reference descriptors:
    //cv::Mat refDescriptors=loadDescriptors(&infile);
    //cv::Mat refImageids=refDescriptors.col(64);
    //refDescriptors=refDescriptors.colRange(0, refDescriptors.cols-1);
    //refDescriptors=refDescriptors.clone();
    
    //Append reference data to 10K random sampled features
//    cv::descriptors_orig.push_back(refDescriptors); //For the full 10K+1

    //Convert input db to Matrix<float> for compatibility with flann
    flann::Matrix<float> descriptors_db=Mat2Float(&descriptors); //For the full 10K+!
    //flann::Matrix<float> descriptors_db=Mat2Float(&refDescriptors);
    
    //cv::Mat descriptors=descriptors_orig.colRange(0, descriptors_orig.cols-1);
    std::cout<<"Clustering vocabulary..."<<std::endl;
    boost::timer::auto_cpu_timer t;
    //cv::Mat vocabulary = hiKMeansCluster(descriptors_db, clusters);
    printf("Allocating Cluster Data...\n");
    flann::Matrix<float> clusterCenters(new float[clusters*descriptors.cols], clusters, descriptors.cols);

    
    flann::KMeansIndexParams kParams = flann::KMeansIndexParams(branches, iterations, flann::FLANN_CENTERS_KMEANSPP,0.2);
    int numClusters =flann::hierarchicalClustering<flann::L2<float> >(descriptors_db, clusterCenters, kParams);
    t.report();
    
    cv::Mat clustersMat(clusters, descriptors_orig.cols,CV_32F);
    for (int i=0;i<clustersMat.rows;i++){
        for (int j=0;j<clustersMat.cols;j++){
            clustersMat.at<float>(i,j)=clusterCenters[i][j];
        }
    }

    printf("\nWriting Vocabulary to the csv file");
    //writeMatrixToCsv(clusterCenters,outfile.c_str());
    cv::FileStorage outfileStorage(outfile.c_str(), cv::FileStorage::WRITE);
    outfileStorage<<"vocabulary"<<clustersMat;
    outfileStorage.release();
    printf("\nDone! Exiting now...\n");
    
    return 0;
    
}

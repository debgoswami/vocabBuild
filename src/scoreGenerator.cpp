//
//  scoreGenerator.cpp
//  SIFT_FeatureGenerator
//
//  Created by Debaditya on 6/5/15.
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

int main(int argc, char** argv){
    
    //Set up variables
    string qfile = argv[2];//Query Input Matrix of Features
    string cfile = argv[4];//Cluster Centroids (visual word vocab)
    string ifile = argv[6];//Inverted File List


    //--------Load Cluster Centroids -----------
    cv::FileStorage clusterInput(cfile.c_str(), cv::FileStorage::READ);
    cv::Mat vocabularyMat;
    clusterInput["vocabulary"]>>vocabularyMat;
    vocabularyMat=vocabularyMat.clone(); //Ensure Contiguous Memory
    clusterInput.release();
    flann::Matrix<float> vocabulary=Mat2Float(&vocabularyMat);
    
    printf("Check Loaded data. Row 1 of vocabulary:\n");
    for(int i=0;i<vocabularyMat.cols;i++){
        printf("%f",vocabulary[0][i]);
    }
    printf("\n");
    
    //--------Load Query Image Features -----------
    printf("Loading query image features\n");
    cv::Mat qdescriptors_orig=loadDescriptors(&qfile);
    cv::Mat qimageids=qdescriptors_orig.col(qdescriptors_orig.cols-1);
    cv::Mat qdescriptors = qdescriptors_orig.colRange(0,qdescriptors_orig.cols-1);
    qdescriptors=qdescriptors.clone();
    qimageids=qimageids.clone();
    flann::Matrix<float> qdescriptors_db=Mat2Float(&qdescriptors);
    printf("Query Data of size: %i x %i\n",qdescriptors_db.rows, qdescriptors_db.cols);

    //-------------Index the clusters ------------
    printf("Index cluster centroids\n");
    flann::KMeansIndexParams kParams = flann::KMeansIndexParams(32, 5, flann::FLANN_CENTERS_RANDOM, 0.200000000298023224);
    flann::Index<flann::L2<float> > index(vocabulary, kParams);
    index.buildIndex();
    printf("Done...\n");



    //-------------Search the Index ------------
    int nn=10; //We'll truncate to the top-1 Match later
    flann::Matrix<int> indices(new int[qdescriptors.rows*nn], qdescriptors.rows, nn);
    flann::Matrix<float> dists(new float[qdescriptors.rows*nn], qdescriptors.rows, nn);
    flann::SearchParams sParams = flann::SearchParams(684); //684 Checks
    sParams.cores=4; //4 Cores
    printf("Begin knnSearch of query to clusters...\n");
    index.knnSearch(qdescriptors_db, indices, dists, nn, sParams);

    //Now Create the list of 
    std::vector<pair<int, float> > MatchedClusterids;

    for (int i=0;i<descriptors_db.rows;i++)
    {
        MatchedClusterids.push_back(std::make_pair(indices[i][0], refImageids.at<float>(i,0)));
    }

    sort(MatchedClusterids.begin(),MatchedClusterids.end()); //Sort the Matched Cluster ids by cluster id





    return 0;
}

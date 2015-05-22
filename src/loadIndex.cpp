//
//  main.cpp
//  loadIndex
//
//  Created by Debaditya on 17/4/15.
//  Copyright (c) 2015 Debaditya. All rights reserved.
//
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <vector>
#include <fstream>
#include <algorithm>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <flann/flann.hpp>
#include <boost/timer/timer.hpp>
#include <cmath>

#include "map_Utils.h"


int main(int argc, const char * argv[]) {
    std::string indexfile="/Users/deb/dev/distractorSURFflannIndex.data";
    std::string queryfile="/Users/deb/dev/starwarsSURFFeature.yml";
    
    //Load query descriptors:
    cv::Mat qdescriptors_orig=loadDescriptors(&queryfile);
    //Print out first row of the qdescriptors_orig
    printf("Printing out first row of descriptors_orig\n");
    for (int i=0;i<qdescriptors_orig.cols;i++)
        std::printf("%f,",qdescriptors_orig.at<float>(0,i));
    printf("\n");
    
    
    
    
    cv::Mat qimageids=qdescriptors_orig.col(64);
    cv::Mat qdescriptors=qdescriptors_orig.colRange(0,qdescriptors_orig.cols-1);
    
    
    
    flann::Matrix<float> qdescriptorsdb=Mat2Float(&qdescriptors);
    printf("Printing out first row of flann::Matrix<float> descriptors which is itself of dimensionalty %d Rows x %zu Columns\n",qdescriptors.rows,qdescriptorsdb.cols);
    for (int i=0;i<qdescriptorsdb.cols;i++)
        printf("%f,",qdescriptorsdb[0][i]);
    printf("\n\n");
    
    //Load the index
    int nn=10;
    flann::Matrix<int> indices(new int[qdescriptorsdb.rows*nn], qdescriptorsdb.rows, nn);
    flann::Matrix<float> dists(new float[qdescriptorsdb.rows*nn], qdescriptorsdb.rows, nn);
    flann::Index<flann::L2<float> > index_data(qdescriptorsdb, flann::SavedIndexParams(indexfile.c_str()));
    
    
    //Query the index
    boost::timer::auto_cpu_timer q;
    index_data.knnSearch(qdescriptorsdb,indices,dists,nn,flann::SearchParams(128));
    q.report();
    return 0;
}

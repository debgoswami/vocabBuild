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
#include <algorithm>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/imgproc/imgproc.hpp>

#include <flann/flann.hpp>
#include <boost/timer/timer.hpp>
#include <cmath>

#include "/Users/deb/dev/Cpp_code/FeatureGenerator/map_Utils.h"
using namespace std;

bool compareFrequency(const std::pair<float, size_t>& lhs, const std::pair<float, size_t>& rhs) {
    return lhs.second < rhs.second;
}

void readme() //Function to display help
{
    std::cout<<" Usage: ./indexFlann -i </full/path/to/inputfeatures.yml> -o </full/path/to/index.data> "<<std::endl;
}
template<int, typename Allocator>
std::map<int, size_t> frequencies(std::vector<int, Allocator> const &src)
{
    std::map<int, size_t> retval;
    for (auto && x:src)
        ++retval[x];
    return retval;
}

struct Int2 {
    int a[2];
};


struct predicate
{
    bool operator()(const std::pair<size_t, float> &left, const std::pair<size_t, float> &right)
    {
        return left.first > right.first;
    }
};


typedef std::map<float, size_t>::iterator iter; //Iterator for sorting through the map values

int main(int argc, char** argv)
{
    if( argc < 3 )
    { readme(); return -1; }
    
    std::cout<<"Instantiating Variables..."<<std::endl;
    //Instantiate Variables
    std::string infile = argv[2];
    std::string queryfile=argv[4];
    std::string outfile = argv[6];
    int cores=std::atoi(argv[7]);
    int nn=std::atoi(argv[8]);
    int checks=std::atoi(argv[9]);

    std::cout<<"Reading vocabulary image descriptors from :"<<infile<<std::endl;
    std::cout<<"Reading query image descriptors from :"<<queryfile<<endl;
    std::cout<<"Index will be stored in: "<<outfile<<std::endl;
    std::cout<<"Number of cores: "<<cores<<std::endl;
    std::cout<<"Number of checks: "<<checks<<std::endl;
    std::cout<<"Number of nn: "<<nn<<std::endl;

    
    
    
    
    //--------------------------------- Load Reference Images   ---------------------------------//
    //Load the distractor descriptors:
    cv::Mat descriptors_orig=loadDescriptors(&infile);
     printf("Data type for descriptors_orig : %d \n",descriptors_orig.type());
    cv::Mat rimageids=descriptors_orig.col(64);
    rimageids=rimageids.clone();
//    rimageids.convertTo(rimageids, CV_8U);
    cv::Mat descriptors=descriptors_orig.colRange(0, descriptors_orig.cols-1);
    //cv::Mat descriptors=descriptors_orig.colRange(0, 31); //Try 32 Feature (just to get a feel for search time
    descriptors=descriptors.clone();
    flann::Matrix<float> descriptors_db=Mat2Float(&descriptors);
    
    
    
    
    
    
    //--------------------------------- Now Build Index   ---------------------------------//
    std::cout<<"Building index..."<<std::endl;
    
    flann::KMeansIndexParams kParams = flann::KMeansIndexParams(32, 5, flann::FLANN_CENTERS_RANDOM,0.20000000298023224);
    //Try the linear brute force search instead as well
    //    flann::CompositeIndexParams indexParams( int trees = 4,int branching = 32, int iterations = 11, flann::flann_centers_init_t centers_init = flann::FLANN_CENTERS_RANDOM, float cb_index = 0.2);
    boost::timer::auto_cpu_timer t;
    flann::Index<flann::L2<float> > index(descriptors_db, kParams);
    index.buildIndex();
    t.report();
    //Store the index
    std::cout<<"Writing Index data to : "<<outfile<<std::endl;
    index.save(outfile.c_str());
    //-----------------------------------------------------------------------------------//
    
    
    
    
    
    
    //--------------------------------- Now Begin Testing on Query Images   ---------------------------------//
    printf("Reading input image list\n");
    //    string queryfile="/Users/deb/dev/Image_data/referenceImgs/imagelist.txt";
    //    std::vector<std::string> qFilelist = readfilelist(queryfile);
    //    cv::Mat qdescriptors_orig=computeSURFDescriptors(qFilelist);
//    string queryfile="/Users/deb/dev/query100SURFDescriptors.yml";
//    string queryfile="/Users/deb/dev/MiniSURFDescriptors.yml";
    cv::Mat qdescriptors_orig=loadDescriptors(&queryfile);
    //printf("Data type for qdescriptors_orig : %d \n",qdescriptors_orig.type());
    cv::Mat qimageids=qdescriptors_orig.col(64);
    qimageids=qimageids.clone();
    cv::Mat qdescriptors=qdescriptors_orig.colRange(0, qdescriptors_orig.cols-1);
    //cv::Mat qdescriptors=qdescriptors_orig.colRange(0, 31); //Try 32 features to test search time
    qdescriptors=qdescriptors.clone();
    flann::Matrix<float> descriptors_q=Mat2Float(&qdescriptors);
    //cout<<"Query Descriptors Zeroth Row : "<<descriptors_q[0][0]<<std::endl;
    
    
    flann::Matrix<int> indices(new int[descriptors_q.rows*nn], descriptors_q.rows, nn);
    flann::Matrix<float> dists(new float[descriptors_q.rows*nn], descriptors_q.rows, nn);
    
    
    printf("\nSearching index\n");
    flann::SearchParams sParams = flann::SearchParams(checks);

    sParams.cores=cores;
    //printf("Number of cores: %d\n",cores);
    //boost::timer::auto_cpu_timer q;
    double tstart = 0.0;
    double tend = 0.0;
    double t_loop = 0.0;
       
    tstart = cv::getTickCount();
    index.knnSearch(descriptors_q, indices, dists, nn, sParams); //Use Radem's values
    tend=cv::getTickCount();
    t_loop = (tend - tstart) / cv::getTickFrequency();
    printf("Time taken for search: %f\n",t_loop);

    //q.report();
    printf("Finished searching...\nProcessing mAP Scores\n");
    
    //Write indices/dists/qimageids/rimageids to .csv
    writeMatrixToCsv(indices, "/Users/deb/dev/benchmarkResults/indices.csv");
    writeMatrixToCsv(dists, "/Users/deb/dev/benchmarkResults/dists.csv");
    
   // std::ofstream qidfile("/Users/deb/dev/benchmarkResults/qimageids.csv",ios::out|ios::trunc);
   // qidfile<<cv::format(qimageids,"csv");
   // qidfile.close();
   // 
   // std::ofstream ridfile("/Users/deb/dev/benchmarkResults/rimageids.csv",ios::out|ios::trunc);
   // ridfile<<cv::format(rimageids,"csv");
   // ridfile.close();
    
//    writeMatrixToCsv(qfloatids, "/Users/deb/dev/benchmarkResults/qimageids.csv");
//    writeMatrixToCsv(rfloatids, "/Users/deb/dev/benchmarkResults/rimageids.csv");
    
    
//    cout<<"Sample indices : "<<endl;

//    for (int p=0;p<nn;p++){
//        printf("Index at [%d ,0]: %d\n",p,indices[p][0]);
//        printf("Index at [%d ,1]: %d\n",p,indices[p][1]);
//        printf("Index at [%d ,2]: %d\n",p,indices[p][2]);
//        printf("Index at [%d ,3]: %d\n",p,indices[p][3]);
//        printf("Reference row 1: %f\n",rimageids.at<float>(indices[p][0],0));
//        printf("Reference row 2: %f\n",rimageids.at<float>(indices[p][1],0));
//        printf("Reference row 3: %f\n",rimageids.at<float>(indices[p][2],0));
//        printf("Reference row 4: %f\n\n",rimageids.at<float>(indices[p][3],0));
//    }
//    
    
    
    
    
    
    
    //---------------------------------     Matching   ---------------------------------//
    
    //Get changepoints
    std::vector<int> changepoints;
    float prev_val=qimageids.at<float>(0,0);
    for (int i=0; i < qimageids.rows; i++){
        if (i==0){
            changepoints.push_back(i);
        }
        else if( qimageids.at<float>(i,0)!= prev_val){
            changepoints.push_back(i);
            prev_val=qimageids.at<float>(i,0);

        }
    }
    changepoints.push_back(qimageids.rows);
//    for (int i=1;i<changepoints.size();i++)
//                printf("Image %d has %d Features\n",i,changepoints[i]- changepoints[i-1]);
//    

    
    
    
    
    //---------------------------------     Indexer Loop [CORE MATCHING FUNC]   ---------------------------------//
    
    //Match Stores
    std::vector<float> match_one_ids;
    std::vector<float> match_all_ids;
    std::vector<float> freq_all_ids;
    
    int counter=0;
    int Top_one_mapCounter=0;
    int Top_ten_mapCounter=0;
    int found_topten_match=0;
    int image_nums=changepoints.size()-2;
    float max_id=0;
    
    
    //Vectors and Maps
    vector<pair<size_t,float> >::iterator x;
    vector<pair<size_t,float> > allFreqVec;
    std::map<float, size_t> histogram;
    std::map<float, size_t> histogram_all;
//    printf("Number of rows in changepoints = %lu",changepoints.size());
    
    //---------------mAP Loop !Danger Will Robinson!--------------//
    for (int i=0;i<image_nums;i++){ //For each image in the query list
        
        //Clear the counters
        match_one_ids.clear();
        match_all_ids.clear();
        max_id=0;
        histogram.clear();
        histogram_all.clear();
        freq_all_ids.clear();

        //Now let's loop over all the points for that image
        for(int p = changepoints[i]; p<changepoints[i+1];p++){ //While the image id is constant
            match_one_ids.push_back(rimageids.at<float>(indices[p][0],0)); //Add the matched image id for 1st row to the vector<int>
            for (int q=0;q<nn;q++){
                match_all_ids.push_back(rimageids.at<float>(indices[p][q],0));
            }
        }
        
        // Sort the matched_ids
        sort(match_one_ids.begin(), match_one_ids.end());
        sort(match_all_ids.begin(),match_all_ids.end());
        
        //Create the map of frequencies for the Top-nn
        for (auto const & x : match_one_ids)
        {
            std::map<float, size_t>::iterator it = histogram.find(x);
            if(it != histogram.end()) {
                //update the count
                it->second += 1;
            }
            else {
                //new occurrence
                histogram.insert(std::make_pair(x, 1));
            }
        }
        
        
        //Sort the histogram for max values
        map<float, size_t>::iterator maxval_iter = std::max_element(histogram.begin(), histogram.end(), compareFrequency);
        max_id=maxval_iter->first;
//        printf("\nTop-nn ID for image #%d = %f\n",i,max_id);

        //Create the histogram of all:

        for (auto const & x : match_all_ids)
        {
            std::map<float, size_t>::iterator it = histogram_all.find(x);
            if(it != histogram_all.end()) {
                //update the count
                it->second += 1;
            }
            else {
                //new occurrence
                histogram_all.insert(std::make_pair(x, 1));
            }
        }


        iter it_all= histogram_all.begin();
        iter end_all= histogram_all.end();
        
        for(;it_all!=end_all;++it_all){
            freq_all_ids.push_back(it_all->first);
            allFreqVec.push_back(std::make_pair(it_all->second, it_all->first));
        }
//        sort(freq_all_ids.begin(),freq_all_ids.end());
        sort(allFreqVec.begin(), allFreqVec.end(), predicate());
        
//        x=allFreqVec.begin();
//        counter=0;
//        for(;counter<nn;++x){
//            printf("image id: %f, count: %lu\n",x->second,x->first);
//            ++counter;
//        }
        
//        std::sort(myVector.begin(), myVector.end(), [](const std::vector< int >& a, const std::vector< int >& b){ return a[1] > b[1]; } ); //If you want to sort in ascending order, then substitute > with <.


        // Counter code to populate the Match scores for One and Ten respectively
        if (max_id == qimageids.at<float>(changepoints[i]+1,0))
            ++Top_one_mapCounter;
       // printf("Top-nn ID for image %f = %f with %lu Matches    ||   ",qimageids.at<float>(changepoints[i],0),max_id,histogram[max_id]);
        //printf("Compare with Matches for image:image : %lu\n",histogram[qimageids.at<float>(changepoints[i],0)]);
//        printf("Top one map score=%d\n+\n",Top_one_mapCounter);


        //For the Ten
        counter=0;
        x=allFreqVec.begin();
        for(;found_topten_match !=0 && counter<nn;x++){
            if(x->second == qimageids.at<float>(changepoints[i],0)){
                ++Top_ten_mapCounter;
                printf("top-10nn Match for image %f : Matched id = %f\n",qimageids.at<float>(changepoints[i],0),x->second);
                printf("Top ten map score=%d\n",Top_ten_mapCounter);
                found_topten_match=1;
            }
            ++counter;
        }
        
    }//End of for onemAP Lop
    
    printf("\nTop 1 mAP Score : %d \n",100*Top_one_mapCounter/image_nums);
    printf("\nTop 10 mAP Score : %d \n\n",100*Top_ten_mapCounter/image_nums);
    
    //    flann::save_to_file(indices,const* "/Users/deb/dev/knnIndicesResult.data","result");
    return 0;
}

//
//  buildVocab.cpp
//  Create an inverted File List from an input matrix of cluster centroids
//
//  Created by Debaditya on 29/4/15.
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
#include <cmath>
#include <set>
#include "map_Utils.h"
using namespace std;
//using namespace cv;

//struct invertedIndexFile 
//{
//    int clusterid; //Cluster id 
//    int num_images;//How many images visited this cluster [for TF IDF]
//    vector<pair<float, int> > imageCounts; //Vector of <imageids/counts>
//};

bool compareIVF(const std::pair<int, float>& lhs, std::pair<int, float>& rhs) {
    return (lhs==rhs);
}

//Predicate to enable sorting by second element of vector<pair<int,float> >
bool bySecondElement(const std::pair<int, float>& lhs, const std::pair<int, float>& rhs){
    return lhs.second < rhs.second;
} 

struct invertedIndexFile 
{
    int clusterid; //Cluster id 
    int num_images;//How many images visited this cluster [for TF IDF]
    std::map<float, int> imageCounts; //Histogram of <imageids/counts>
};

int main(int argc, char** argv)
{
	//Instantiate Arguments
	std::string clusterfile = argv[2];
	std::string queryfile = argv[4];
	std::string invertedfile =argv[6];


	//Load the cluster centroids matrix
	cv::FileStorage readClusters(clusterfile, cv::FileStorage::READ);
	cv::Mat clusters;
	readClusters["vocabulary"]>>clusters;
	readClusters.release();
	clusters=clusters.clone();
    flann::Matrix<float> clusters_db=Mat2Float(&clusters);


	//Load the image descriptors
	cv::Mat refdescriptors_orig=loadDescriptors(&queryfile);
	cv::Mat refImageids=refdescriptors_orig.col(refdescriptors_orig.cols-1);
    refImageids=refImageids.clone();
    cv::Mat refdescriptors = refdescriptors_orig.colRange(0,refdescriptors_orig.cols-1);
    refdescriptors=refdescriptors.clone();
    flann::Matrix<float> descriptors_db=Mat2Float(&refdescriptors);


	//Build index of cluster centroids
    //std::cout<<"Building index..."<<std::endl;
    flann::KMeansIndexParams kParams = flann::KMeansIndexParams(32, 5, flann::FLANN_CENTERS_RANDOM,0.20000000298023224);
    flann::Index<flann::L2<float> > index(clusters_db, kParams);
    index.buildIndex();



	//knnSearch to get cluster ids for reference images
	int nn=10; //Placeholder to get the top-10 nn. Will discard 9, and only keep top-most
    int checks = 684; //0.7 Precision
    int cores=4; //4-core parallel search
    flann::Matrix<int> indices(new int[descriptors_db.rows*nn],descriptors_db.rows, nn);
    flann::Matrix<float> dists(new float[descriptors_db.rows*nn],descriptors_db.rows, nn);
    flann::SearchParams sParams=flann::SearchParams(checks);
    sParams.cores=cores;

    //Get timers
    double tstart=0.0;
    double tend=0.0;
    double tloop=0.0;

    //Start Indexing
    tstart=cv::getTickCount();
    index.knnSearch(descriptors_db, indices, dists, nn, sParams);
    tend=cv::getTickCount();
    tloop=(tend - tstart)/cv::getTickFrequency();
    //printf("Search Time : %f\n",tloop);

    std::vector<pair<int, float> > MatchedClusterids;
    //Build the Cluster Histograms
    //For each feature, store the corresponding cluster id (Hard Assignment) and image id (2d vector)
    for (int i=0;i<descriptors_db.rows;i++)
    {
        MatchedClusterids.push_back(std::make_pair(indices[i][0], refImageids.at<float>(i,0)));
    }

    sort(MatchedClusterids.begin(),MatchedClusterids.end()); //Sort the Matched Cluster ids by cluster id

    //save the clustered matrices as a csv for safety and pythonification of results
   // std::fstream matchfile("/home/cuda/deb/MatchedClusterids.csv",std::ios::out | std::ios::binary);
   // for(int j=0;j<MatchedClusterids.size();j++){
   //     matchfile.write((char *)&MatchedClusterids[j].first,sizeof(MatchedClusterids[j].first));
   //     matchfile.write(",",sizeof(","));
   //     matchfile.write((char *)&MatchedClusterids[j].second,sizeof(MatchedClusterids[j].second));
   //     matchfile.write("\n",sizeof("\n"));
   // }
   // matchfile.close();
   
    //Find the changepoints for each clusterid
    std::vector<int> changepoints;
    int prev_val=MatchedClusterids[0].first;
    for (int i=1;i<MatchedClusterids.size();i++) {
        if(MatchedClusterids[i].first != prev_val){
            changepoints.push_back(i);
            prev_val=MatchedClusterids[i].first;
        }
    }
    changepoints.push_back(MatchedClusterids.size());
    
    //Now we can loop through each cluster's worth of imageids
    //  Set up the variables
    int img_count=0; //Counter for an image within ONE cluster
    float img_id=0.0; //Image id being considered
    //std::set<float> unique_imgs; //Set of unique image ids for ONE cluster
    std::map<float, int> img_freqs; //Histogram of image counts for ONE cluster
    std::vector<invertedIndexFile> indexFile;
    invertedIndexFile thisClusterIndexFile;

    int begin,end;
    //--Begin Looping
    for(int j=0;j<changepoints.size()-1;j++){
        begin=changepoints[j];          //Start index for this cluster
        end=changepoints[j+1];          //End index 
        //unique_imgs.clear();          //Clear the set of unique image ids
        
        //Note that in sort, the 'end' iterator element is not accessed. This is why we don't need to search to changepoints(end)-1
        std::sort(MatchedClusterids.begin()+begin, MatchedClusterids.begin()+end, bySecondElement); //Sort the specific cluster id by image id
        //--Now we can populate the histogram
        for (int p=begin;p<end;p++){
            img_id=MatchedClusterids[p].second;
            std::map<float, int>::iterator it = img_freqs.find(img_id);
            if( it !=img_freqs.end()){
                it->second +=1; //Update the count
            }
            else {
                img_freqs.insert(std::make_pair(img_id,1)); //Make a new occurence
            }
        } //End of histogram population for this cluster id
        thisClusterIndexFile.clusterid = MatchedClusterids[begin].first; //Populate the cluster id
        thisClusterIndexFile.num_images = img_freqs.size(); //Number of images visited this cluster
        thisClusterIndexFile.imageCounts = img_freqs; //The map of <imageid, count>

        indexFile.push_back(thisClusterIndexFile); //Append the above struct to the vector
        img_freqs.clear();
    }//End loop
    
    //Now save the inverted index file
    //BOOST Serialization Implementation [Bank for now]

    //Print to Terminal and just redirect to a csv file. Easier that way
    printf("clusterid, num_images\t,[img_id,img_count]\n");
    for(int i=0;i<indexFile.size();i++){
        printf("%i, %i \t",indexFile[i].clusterid,indexFile[i].num_images);
        std::map<float, int>::iterator map_it=indexFile[i].imageCounts.begin();
        for(;map_it!=indexFile[i].imageCounts.end();map_it++){
            printf(", %f, %i",map_it->first, map_it->second);
        }
        cout<<std::endl;
    }

    
    
    
    
    

    return 0;
}

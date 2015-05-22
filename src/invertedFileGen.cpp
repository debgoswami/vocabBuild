//
//  invertedFileGen.cpp
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


#include <cmath>

using namespace std;
//using namespace cv;

//Predicate to enable sorting by second element of vector<pair<int,float> >
bool bySecondElement(const std::pair<int, float>& lhs, const std::pair<int, float>& rhs){
    return lhs.second < rhs.second;
} 

struct invertedIndexFile 
{
    int clusterid; //Cluster id 
    int num_images;//How many images visited this cluster [for TF IDF]
    std::map<float, int> > imageCounts; //Histogram of <imageids/counts>
};

bool compareIVF(const std::pair<int, float>& lhs, std::pair<int, float>& rhs) {
    return (lhs==rhs)
}

int main(int argc, char** argv)
{
    string infile = argv[2];//Load the match matrix
    string outfile = argv[4];//Store the inverted file list
    string qfile = argv[6]; //Load the query image features

    //Load the match matrix into a vector<pair<int, double> >
    std::vector<std::pair<int, float> > MatchedClusterIds;
    std::ifstream loadMatches(infile.c_str());
    MatchedClusterids<<loadMatches.getlines();
    loadMatches.close();
    


    //Find the changepoints for each clusterid
    std::vector<int> changepoints=0;
    int prev_val=MatchClusterIds[0].first;
    for (int i=1;i<MatchedClusterIds.rows;i++) {
        if(MatchedClusterIds[i].first != prev_val){
            changepoints.push_back(i);
            prev_val=MatchedClusterIds[i].first;
        }
    }
    changepoints.push_back(MatchedClusterIds.rows);
    
    //Now we can loop through each cluster's worth of imageids
    //  Set up the variables
    int img_count=0; //Counter for an image within ONE cluster
    float img_id=0.0; //Image id being considered
    //std::set<float> unique_imgs; //Set of unique image ids for ONE cluster
    std::map<float, int> > img_freqs; //Histogram of image counts for ONE cluster
    std::vector<invertedIndexFile> indexFile;
    invertedIndexFile thisClusterIndexFile;

    int begin,end;
    //--Begin Looping
    for(int j=0;j<changepoints.rows-1;j++){
        begin=changepoints[i];          //Start index for this cluster
        end=changepoints[i+1];          //End index 
        //unique_imgs.clear();          //Clear the set of unique image ids
        img_freqs.clear();
        //Note that in sort, the 'end' iterator element is not accessed. This is why we don't need to search to changepoints(end)-1
        std::sort(MatchedClusterIds.begin()+begin, MatchedClusterIds.begin()+end, bySecondElement); //Sort the specific cluster id by image id
        //--Now we can populate the histogram
        for (int p=begin;p<end;p++){
            img_id=MatchedClusterIds[p].second;
            std::map<float, int>::iterator it = img_freqs.find(img_id);
            if( it !=img_freqs.end()){
                it->second +=1; //Update the count
            }
            else {
                img_freqs.insert(std::make_pair(img_id,1)); //Make a new occurence
            }
        } //End of histogram population for this cluster id
        thisClusterIndexFile.clusterid = MatchedClusterIds[begin].first; //Populate the cluster id
        thisClusterIndexFile.num_images = img_freqs.size(); //Number of images visited this cluster
        thisClusterIndexFile.imageCounts = img_freqs; //The map of <imageid, count>

        indexFile.push_back(thisClusterIndexFile); //Append the above struct to the vector
    }//End loop
    
    //Now save the inverted index file
    //BOOST Serialization Implementation 

    //--Get the cluster ids for the query image
    

    return 0;
}

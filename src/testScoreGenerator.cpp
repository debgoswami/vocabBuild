//
//  testScoreGenerator.cpp
//  Create an inverted File List from an input matrix of cluster centroids
//
//  Created by Debaditya on 29/4/15.
//  Copyright (c) 2015 Debaditya. All rights reserved.
// This program runs the entire vocabBuilder.cpp code and then simply queries the images. This is done to ease off the loading functions.
// Changes can be made to replace the first part simply with the invertedFileList Reader function

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

bool compareIVF(const std::pair<int, float>& lhs, std::pair<int, float>& rhs) {
    return (lhs==rhs);
}

//Predicate to sort a map<flaot, size_t> by second element
//bool compareFrequency(const std::pair<float, size_t>& lhs, const std::pair<float, size_t>& rhs) {
//    return lhs.second < rhs.second;
//}
bool compareFrequency(const std::pair<float, float>& lhs, const std::pair<float, float>& rhs) {
    return lhs.second < rhs.second;
}
bool compareFrequency_desc(const std::pair<float, float>& lhs, const std::pair<float, float>& rhs) {
    return lhs.second > rhs.second;
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

//Predicate to search through a vector<invertedIndexFile> by clusterid
struct find_clusterid : std::unary_function<invertedIndexFile, bool> {
    int clusterid;
    find_clusterid(int clusterid):clusterid(clusterid) { }
    bool operator()(invertedIndexFile const& ivf) const {
        return ivf.clusterid==clusterid;
    }
};


std::map<float, int> getFeatureCounts(const cv::Mat &img_ids)
{
    //Find the changepoints for each clusterid
    std::map<float, int> imgFeatures;
    for (int i=0;i<img_ids.rows;i++) {
        ++imgFeatures[img_ids.at<float>(i,0)];
            }
    return imgFeatures;
}

//
//------------------------                  MAIN FUNCTION               -----------------------------
//


int main(int argc, char** argv)
{
	//Instantiate Arguments
	printf("Initialising...");
	std::string clusterfile = argv[2];
	std::string queryfile = argv[4];
	std::string infile =argv[6];
	printf("Done\n");


	//Load the cluster centroids matrix
	printf("Loading in vocabulary...");
	cv::FileStorage readClusters(clusterfile, cv::FileStorage::READ);
	cv::Mat clusters;
	readClusters["vocabulary"]>>clusters;
	readClusters.release();
	clusters=clusters.colRange(0,clusters.cols-1);
	clusters=clusters.clone();
    flann::Matrix<float> clusters_db=Mat2Float(&clusters);
    printf("Done\n");


	//Load the image descriptors
	printf("Reading reference Dataset...");
	cv::Mat refdescriptors_orig=loadDescriptors(&queryfile);
	cv::Mat refImageids=refdescriptors_orig.col(refdescriptors_orig.cols-1);
    refImageids=refImageids.clone();
    cv::Mat refdescriptors = refdescriptors_orig.colRange(0,refdescriptors_orig.cols-1);
    refdescriptors=refdescriptors.clone();
    flann::Matrix<float> descriptors_db=Mat2Float(&refdescriptors);
    printf("Done\n");
    printf("Getting reference Image Descriptors...");
    std::map<float, int> rImageCounts = getFeatureCounts(refImageids);  //Store the image Features for the vocabulary
    printf("Done\n");


	//Build index of cluster centroids
    //std::cout<<"Building index..."<<std::endl;
    printf("Building index...");
    flann::KMeansIndexParams kParams = flann::KMeansIndexParams(32, 5, flann::FLANN_CENTERS_RANDOM,0.20000000298023224);
    flann::Index<flann::L2<float> > index(clusters_db, kParams);
    index.buildIndex();
    printf("Done\n");



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
    printf("Searching through index...");
    tstart=cv::getTickCount();
    index.knnSearch(descriptors_db, indices, dists, nn, sParams);
    tend=cv::getTickCount();
    tloop=(tend - tstart)/cv::getTickFrequency();
    printf("Search Time : %f\n",tloop);

    std::vector<pair<int, float> > MatchedClusterids;
    
    
    
    
    
    //------------------- Build the Cluster Histograms --------------------



    printf("Building histograms...");


    //For each feature, store the corresponding cluster id (Hard Assignment) and image id (2d vector)
    for (int i=0;i<descriptors_db.rows;i++)
    {
        MatchedClusterids.push_back(std::make_pair(indices[i][0], refImageids.at<float>(i,0)));
    }

    sort(MatchedClusterids.begin(),MatchedClusterids.end()); //Sort the Matched Cluster ids by cluster id

    //Write the matched histogram into a file
    std::vector<pair<int, float> >::iterator matcherit=MatchedClusterids.begin();
    ofstream matcherfile("/home/cuda/deb/refMatchedIds.csv");
    for (;matcherit != MatchedClusterids.end();matcherit++)
    {
        matcherfile<<matcherit->first<<","<<matcherit->second<<"\n";
    }
    matcherfile.close();
     
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

    printf("Generating Inverted File List...");
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
    printf("Done\n");

    //Clear Memory
    printf("Clearing Memory\n");
    //delete[] descriptors_db.ptr();
    //printf("descriptors_db, ");
    refdescriptors.release();
    printf("refdescriptors, ");
    delete[] indices.ptr();
    printf("indices, ");
    delete[] dists.ptr();
    printf("dists\n");
    




   //-----------------                  Now begin Querying Images               -----------------------//
   
   
   
   
   printf("Begin Query Image Processing\nLoading Images...");
    cv::Mat qdescriptors_orig=loadDescriptors(&infile);
    cv::Mat qImageids=qdescriptors_orig.col(qdescriptors_orig.cols-1);
    qImageids=qImageids.clone();
    cv::Mat qdescriptors = qdescriptors_orig.colRange(0,qdescriptors_orig.cols-1);
    qdescriptors=qdescriptors.clone();
    flann::Matrix<float> qdescriptors_db = Mat2Float(&qdescriptors);
    printf("Done\n");


    //knnSearch through the images
    printf("Searching through the index for the QUERY Images...");
    flann::Matrix<int> qindices(new int[qdescriptors_db.rows*nn],qdescriptors_db.rows, nn);
    flann::Matrix<float> qdists(new float[qdescriptors_db.rows*nn],qdescriptors_db.rows, nn);
    index.knnSearch(qdescriptors_db,qindices,qdists,nn,sParams);
    printf("Done\n");

    printf("Calculating number of input images and their changepoints...");
    changepoints.clear();
    changepoints.push_back(0);
    float qprev_val=qImageids.at<float>(0,0);
    for (int i=0;i<qImageids.rows;i++){
        if(qImageids.at<float>(i,0) != qprev_val)
        {
            changepoints.push_back(i);
            qprev_val=qImageids.at<float>(i,0);
        }
    }
    changepoints.push_back(qImageids.rows);
    printf("Done. %lu Images to be queried\n",changepoints.size()-1);








    //Instantiate Score Data Structures
    std::vector<int> match_one_ids;                                 //Top one matched cluster ids for the query image
    std::map<int, size_t> histogram;                                //Histogram of cluster counts for each query image
    int img_nums = changepoints.size()-1;
    float map_one_score=0.0;                                        //Counter to track mAP score
    float map_ten_score=0.0;                                        //Counter to track ten mAP score
    float max_id=0.0;                                               //Matched img id returned after scoring
    int feature_nums;                                               //Features for the query Image
    float imgcount_score;                                           //TFIDF Score for a particular ref image within the query image loop
    int termCount;                                                  //How many times the query image visited THIS clusterid
    int fNums;                                                      // Number of features for this reference image
    std::map<float, size_t> imageScores;                            //Map of <ref_imageid, count> [Deprecated]
    std::map<float, float> imageScores_normalised;                  //Map of <ref_imageid, score> [TFIDF Scores]
    std::vector<std::pair<float, float> > top_ten_scores;                     //Vector to store the above Map into a sortable structure
    int found=0;



    //------------------------          CALCULATE SCORES/HISTOGRAMS         -------------------------//
    



    printf("Calculating scores....\n");

    for (int i=0;i<img_nums;i++){
        feature_nums=changepoints[i+1]-changepoints[i];
        printf("Image %i has %i features\n",i,feature_nums);
        match_one_ids.clear();
        histogram.clear();
        max_id=0.0;

        for(int p=changepoints[i];p<changepoints[i+1];p++)
        {
            match_one_ids.push_back(qindices[p][0]);

        }//Looping over features for ONE Image
        sort(match_one_ids.begin(), match_one_ids.end());

        //Create a histogram of <int cluster_id, size_t count> for THIS image
        for (auto const & x : match_one_ids)
        {
            std::map<int, size_t>::iterator it = histogram.find(x);
            if(it != histogram.end()) {
                it->second+=1;
            }
            else {
                histogram.insert(std::make_pair(x,1));
            }
        }//End of histogram of cluster_ids (Sorted)
        //if (i==671){
        //    std::cout<<"Image: "<<refImageids.at<float>(i,0)<<" histogram\n"<<std::endl;
        //    std::cout<<histogram<<std::endl;"
        //}
        //printf("Writing histogram of cluster_ids to file: ~/deb/query100_MatchResults.csv\n");
        std::ofstream matchfile;
        matchfile.open("/home/cuda/deb/query100_MatchResults.csv", ios::app);
        for (auto const & x : histogram)
        {
            matchfile<<qImageids.at<float>(changepoints[i])<<","<<x.first<<","<<x.second<<std::endl;

        }
        matchfile.close();

        //----------------------------------------------------------------------------------------
        //--NOTE:
        //      We now have a histogram of clusters and their counts for this image.
        //      To calculate the best image match in our db for this query image:
        //          - For EACH clusterid in our histogram
        //              - Find the corresponding clusterid row in our inverted file list
        //              - For EACH image in that cluster
        //                  - Create a float img_score (TF-IDF Weighted OR just freq-based)
        //                      - Freq-based calculation = cluster_count/feature_nums * img_count
        //                      - TF-IDF = cluster_count/feature_nums * log(Vocabsize/term_count)
        //                  - Append the score to a map of type <float img_id, float img_score>
        //          - Return the img_id for max(float img_score)
        //          - IF (img_id == query_img_id)
        //              - mAP_one_score++;
        //
        //--END-NOTE
        //----------------------------------------------------------------------------------------
        imageScores.clear();
        imageScores_normalised.clear();
        std::map<int, size_t>::iterator it = histogram.begin();
        for (;it != histogram.end();it++) //For each clusterid in THIS qimage
        {
            std::vector<invertedIndexFile>::iterator thisCluster=std::find_if(indexFile.begin(), indexFile.end(), find_clusterid(it->first));
            std::map<float, int>::iterator  thisClusterImages = thisCluster->imageCounts.begin();                       //Points to one <image, count> pair within the map for THIS clusterid
            while(thisCluster->num_images !=0 && thisClusterImages != thisCluster->imageCounts.end())
            {
                std::map<float, float>::iterator imgScore_it = imageScores_normalised.find(thisClusterImages->first);   //imgScore_it points to refImage within the global histogram of <img,scores> for THIS query image
                

                //----------------------
                //CALCULATE TF_IDF SCORE
                //----------------------
                termCount = float(it->second);                      //How many times the query image visited THIS clusterid
                fNums = rImageCounts[thisClusterImages->first];     // Number of features for this reference image
                //imgcount_score = (float(it->second) / float(feature_nums))* log(1000/thisClusterImages->second);
                imgcount_score = (float(thisClusterImages->second) / float(fNums))* log(1000/thisCluster->num_images); //

                if (fNums < 250 && (thisClusterImages->first != qImageids.at<float>(changepoints[i]+1,0)))
                    imgcount_score=0;
                
                if (imgScore_it != imageScores_normalised.end())
                {
                    imgScore_it->second+=imgcount_score;

                }
                else
                {
                    imageScores_normalised.insert(std::make_pair(thisClusterImages->first, imgcount_score));
                }
                ++thisClusterImages;
            }//End looping over images for ONE cluster
        }
        
        //Now Sort the histogram for max Values by count (i.e. top-most occuring imageid)
        //std::map<float, size_t>::iterator maxval_iter = std::max_element(imageScores.begin(), imageScores.end(), compareFrequency);
        std::map<float, float>::iterator maxval_iter = std::max_element(imageScores_normalised.begin(), imageScores_normalised.end(), compareFrequency);
        max_id = maxval_iter->first;

        top_ten_scores.clear();                 //Clear the vector to store the <img,score> values
        //Now Build the top-10 histogram
        for (auto x=imageScores_normalised.begin(); x != imageScores_normalised.end(); x++)
        {
            top_ten_scores.push_back(std::make_pair(x->first, x->second));
        }
        sort(top_ten_scores.begin(), top_ten_scores.end(), compareFrequency_desc);

        printf("qimg: %f -> %f [%f Counts]",qImageids.at<float>(changepoints[i],0), max_id, maxval_iter->second);
        printf(" Expected match counts = [%f]\n",imageScores_normalised[qImageids.at<float>(changepoints[i],0)]);

        printf("Top 10 Matches: [ ");
        for (int s=0;s<10;s++)
        {
            printf("(%f,%f) ",top_ten_scores[s].first, top_ten_scores[s].second);
           if (top_ten_scores[s].first == qImageids.at<float>(changepoints[i]+1,0))
           {
               ++map_ten_score;
           }
        }
        printf("]\n");

        if (max_id == qImageids.at<float>(changepoints[i]+1,0))
            ++map_one_score;

    }//End of global per/image loop

    printf("Top ONE mAP Score: %f\n", map_one_score);
    printf("Top TEN mAP Score: %f\n", map_ten_score);

    return 0;
}

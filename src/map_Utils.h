//
//  Header.h
//  FeatureGenerator
//
//  Created by Debaditya on 13/4/15.
//  Copyright (c) 2015 Debaditya. All rights reserved.
//

#ifndef FeatureGenerator_Header_h
#define FeatureGenerator_Header_h
#endif

#include <stdio.h>
#include <iostream>
#include "global.h"
#include <fstream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <flann/flann.hpp>


using namespace std;



cv::Mat extractSift(char *pathToTxtFile, const int numImagesToTrain)
{
    /**
     *  This function reads the first 'numImagesToTrain' images whose paths are given in text files. The image read is resized to a standard size.
     *  The keypoints are detected using SiftFeatureDetector( MAXIMAL_KEYPOINTS ) and described using SiftDescriptorExtractor.
     *  A Mat of size NumberofFeatures x FeatureDimensions is returned. Using SIFT, this function returns a Mat of size NumberofFeatures x 128.
     *
     *  Parameter: #  pathToTxtFile - path to a directory which contains text files. Each text file corresponds to one class and contains path to images.
     *             #  numImagesToTrain - Maximal number of images to read from each txt files. First 'numImagesToTrain' many images are read.
     **/

    if(DISPLAY)
    {
        namedWindow("img", cv::WINDOW_NORMAL );
    }

    cv::Mat siftFeature;

    int imgCounter=0;

    for(int i=0;i<6;i++)
    {
        char txtFileName[200];
        sprintf(txtFileName,"%s/%s.txt", pathToTxtFile, txtFiles[i]);
        FILE *filePointer = fopen(txtFileName,"r");
        printf("\n*******************************************************************");
        printf("\nReading %s", txtFileName);

        if(filePointer==NULL)
        {
            printf("\nERROR..!! File '%s' not found", txtFileName);
            return cv::Mat();
        }

        while(imgCounter < numImagesToTrain && !feof(filePointer))
        {
            char pathToImage[500];
            fscanf(filePointer,"%s",pathToImage);

            cv::Mat img = cv::imread(pathToImage,0);
            printf("\nReading : %s",pathToImage);

            if(!img.data)
            {
                printf("\nCould not find %s",pathToImage);
                continue;
            }

            if(img.cols>WIDTH)
            {
                resize(img,img,cv::Size((int)WIDTH,(int)(img.rows*WIDTH/img.cols)));
            }

            if(DISPLAY)
            {
                imshow("img", img);
                cv::waitKey(1);
            }

            cv::Mat descriptor;
            vector<cv::KeyPoint> keypoints;

            detector.detect(img, keypoints);
            extract.compute(img, keypoints,descriptor);
            siftFeature.push_back(descriptor);
            imgCounter++;
        }

        imgCounter = 0;
        fclose(filePointer);
        printf("\n*******************************************************************");
    }

    return siftFeature;
}
cv::Mat Float2Mat(const flann::Matrix<float>* M)
{
    cv::Mat mat_M(M->rows, M->cols, CV_32F, &M);
    return mat_M;
    
}

cv::Mat kMeansCluster(cv::Mat &data,int clusterSize)
{
    /**
     *  implements k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
     *
     *  Parameters: # data- Each sample in rows. Type should be CV_32FC1
     *              # clusterSize- Number of clusters.
     **/
    
    cv::TermCriteria termCriteria(CV_TERMCRIT_ITER,100,0.01);
    int nAttempts=3;
    int flags=cv::KMEANS_PP_CENTERS;
    
    cv::Mat clusterCenters, temp;
    kmeans(data, clusterSize, temp, termCriteria, nAttempts, flags, clusterCenters);
    clusterCenters.convertTo(clusterCenters,CV_8U);
    printf("here\n");
    
    return clusterCenters;
}

//cv::Mat hiKMeansCluster(cv::Mat &data,int clusterSize)
cv::Mat hiKMeansCluster(flann::Matrix<float> &data,int clusterSize)
{
    /**
     *  implements Hierarchical k-means clustering algorithm. The function returns the cluster centers of type CV_8UC1.
     *
     *  Parameters: # data- Each sample in rows. Type should be CV_32FC1.
     *              # clusterSize- Number of clusters.
     **/
    
    cv::Mat matCentres(clusterSize,data.cols,CV_32F);
    flann::Matrix<float> clusterCenters((float*)matCentres.data, matCentres.rows,matCentres.cols);
    
    flann::KMeansIndexParams kParams = flann::KMeansIndexParams(2, 100*clusterSize, flann::FLANN_CENTERS_KMEANSPP,0.2);
    //    int numClusters = flann::hierarchicalClustering<flann::L2<float> >(data, clusterCenters, kParams);
    int numClusters =flann::hierarchicalClustering<flann::L2<float> >(data, clusterCenters, kParams);
    
    
    cv::Mat mat_M(clusterCenters.rows, clusterCenters.cols, CV_32F, &clusterCenters);
    //        mat_M = mat_M.rowRange(cv::Range(0,numClusters));
    //        mat_M.convertTo(mat_M,CV_8U);
    //        cv::Mat clust_centres = cv::Mat(clusterCenters, clusterCenters.cols,clusterCenters.rows,CV_32F);
    return mat_M;
}


template <typename T>
void writeMatrixToCsv(flann::Matrix<T> &m, const char *outfile) {
    std::fstream ofile(outfile,ios::out|ios::in|ios::trunc);
    
    if(!ofile.is_open()){
        printf("Could not open file for writing: %s\n",outfile);
    }
    else{
        printf("Writing into file: %s\n",outfile);
        for (int i=0;i<m.rows;i++)
        {
            for(int j=0;j<m.cols;j++)
            {
                ofile<<m[i][j]<<",";
            }
            ofile<<"\n";
        }
        printf("Finished Writing\n");
    }
    ofile.close();
}

void writeToYMLFile(cv::Mat &dataToWrite, char *fileName)
{
    /**
     *  writes the data to the .yml file.
     *
     *  Parameters: # dataToWrite - matrix to write in .yml file.
     *              # fileName - name of yml file.
     **/
    
    char ymlFileName[100];
    sprintf(ymlFileName,"%s.yml",fileName);
    cv::FileStorage fileStorage(ymlFileName, cv::FileStorage::WRITE);
    
    fileStorage << fileName << dataToWrite;
    
    fileStorage.release();
}


void writeToBinaryFile(cv::Mat &dataToWrite , char *fileName)
{
    /**
     *  writes the data to the .bin file .
     *
     *  Parameters: # dataToWrite - matrix to write in binbary file. Type CV_8UC1.
     *              # fileName - name of binary file.
     **/
    
    fstream binaryFile(fileName,ios::binary|ios::out);
    if(!binaryFile.is_open())
    {
        printf("\nerror in opening: %s", fileName);
        return;
    }
    
    binaryFile.write((char *)dataToWrite.data, dataToWrite.rows*dataToWrite.cols) ;
    
    binaryFile.close();
}


cv::Mat getBowHist(cv::Mat &vocabulary,char *pathToTxtFile, const int numImagesToTrain)
{
    /**
     *  0. Each image is indexed (0-based) in same order as they are read from txt files.
     *  1. This function compute Bag of Words histogram for each image read from txt files. It returns a Mat of size NumberofImagesRead x VocabularySize and type CV_32FC1.
     *  2. The function also writes label of each file into labels.txt file. line i in labels.txt is label for ith image.
     *  3. It also writes location of keypoints of each image into ($imgIndex).txt . Each line in this file is a 3-tuple (vocab-id, x, y) where
     *     keypoint at location (x,y) in image $imgIndex has been assigned to visual word 'vocab-id'.
     *
     *  Parameters: # vocabulary- visual words. Size ClusterSizex128, type CV_8UC1.
     *              # pathToTxtFile - path to a directory which contains text files. Each text file corresponds to one class and contains path to images.
     *              # numImagesToTrain - Maximal number of images to read from each txt files. First 'numImagesToTrain' many images are read.
     **/
    
    if(DISPLAY)
    {
        namedWindow("img", cv::WINDOW_NORMAL );
    }
    
    cv::BOWImgDescriptorExtractor bowDE(new cv::SiftDescriptorExtractor(),new cv::FlannBasedMatcher());
    vocabulary.convertTo(vocabulary,CV_32F);                //convert vocabulary fron CV_8U to CV_32F
    bowDE.setVocabulary(vocabulary);
    
    cv::Mat allHist;
    
    int imgCounter = 0;
    
    int imgIndex=0;
    
    FILE *labelFilePointer=fopen("labels.txt","w");
    
    if(labelFilePointer==NULL)
    {
        printf("\nERROR..!! Couldn't open 'Labels.txt'");
        return cv::Mat();
    }
    
    for(int i=0;i<6;i++)
    {
        
        char txtFileName[200];
        sprintf(txtFileName,"%s/%s.txt",pathToTxtFile,txtFiles[i]);
        FILE *filePointer = fopen(txtFileName, "r");
        printf("\n*******************************************************************");
        printf("\nReading %s", txtFileName);
        
        if(filePointer==NULL)
        {
            printf("\nERROR..!! File '%s' not found", txtFileName);
            return cv::Mat();
        }
        
        while(imgCounter<numImagesToTrain && !feof(filePointer))
        {
            
            fprintf(labelFilePointer,"%d\n",i);
            char pathToImage[500];
            fscanf(filePointer,"%s",pathToImage);//read the pathToImage of ith file
            
            cv::Mat img = cv::imread(pathToImage,0);
            printf("\nReading : %s",pathToImage);
            
            if(!img.data)
            {
                printf("\nCould not find %s",pathToImage);
                continue;
            }
            
            if(img.cols>WIDTH)
            {
                resize(img,img,cv::Size((int)WIDTH,(int)(img.rows*WIDTH/img.cols)));
            }
            
            if(DISPLAY)
            {
                cv::imshow("img", img);
                cv::waitKey(1);
            }
            
            vector<cv::KeyPoint> keypoints;
            vector<vector<int> > pointIdxsOfClusters;
            cv::Mat imgHistogram;
            
            detector.detect(img, keypoints);
            bowDE.compute(img, keypoints,imgHistogram,&pointIdxsOfClusters);
            
            char keypointFileName[200];
            sprintf(keypointFileName,"%s/%d.txt", KEYPOINTS_DIRECTORY,imgIndex);
            
            FILE *keypointFile=fopen(keypointFileName,"w");
            
            for(int k=0;k<pointIdxsOfClusters.size();k++)
            {
                for(int j=0;j<pointIdxsOfClusters[k].size();j++)
                {
                    fprintf(keypointFile,"%d %d %d\n",k, (int)keypoints[pointIdxsOfClusters[k][j]].pt.x, (int)keypoints[pointIdxsOfClusters[k][j]].pt.y);
                }
            }
            
            fclose(keypointFile);
            
            imgIndex++;
            imgCounter++;
            allHist.push_back(imgHistogram);
        }
        
        imgCounter=0;
        fclose(filePointer);
        printf("\n*******************************************************************");
    }
    
    fclose(labelFilePointer);
    return allHist;
}

cv::Mat tfIdfWeighting(cv::Mat &allHist)
{
    /**
     *  This function perform 'term frequency-inverse document frequency' (tf-idf) weighting.
     *  It returns a Mat of size allHist.rows x allHist.cols and type CV_32FC1.
     *
     *  Parameter: # allHist- contains the histogram of all images. Type CV_32FC1.
     **/
    
    cv::Mat weightedAllHist= cv::Mat::zeros(allHist.rows, allHist.cols, CV_32F);
    
    int *numImagesInDb = new int[allHist.cols];
    for(int j=0;j<allHist.cols;j++)
    {
        numImagesInDb[j]=0;
    }
    
    for(int i=0;i<allHist.rows;i++)
    {
        for(int j=0;j<allHist.cols;j++)
        {
            if(allHist.at<float>(i,j)>0)
            {
                numImagesInDb[j]=numImagesInDb[j] + 1;
            }
        }
    }
    
    for(int i=0;i<allHist.rows;i++)
    {
        for(int j=0;j<allHist.cols;j++)
        {
            if(numImagesInDb[j] > 0)
            {
                weightedAllHist.at<float>(i,j)=allHist.at<float>(i,j)*log(((float)(allHist.rows))/numImagesInDb[j]);
            }
            else
            {
                weightedAllHist.at<float>(i,j)=allHist.at<float>(i,j);
            }
        }
    }
    
    delete[] numImagesInDb;
    
    return   weightedAllHist;
}

void writeToBinaryFile(vector<invertedIndex> allIndex , char *fileName)
{
    /**
     *  writes the inverted index into binary file
     *
     *  Parameter: * allIndex- vector of invertedIndex type to write in .bin file.
     *             * fileToWrite- name of file to which vector is to write in .bin file
     **/
    
    fstream binaryFile(fileName, ios::out | ios::binary);
    if(!binaryFile.is_open())
    {
        printf("\nerror in opening: %s", fileName);
        return;
    }
    
    for(int i=0;i<allIndex.size();i++)
    {
        
        for(int j=0;j<allIndex[i].imgIndex.size();j++)
        {
            int imgIndex = allIndex[i].imgIndex[j];
            float weightedHistValue = allIndex[i].weightedHistValue[j];
            binaryFile.write((char *)&imgIndex,sizeof(imgIndex)) ;
            binaryFile.write((char *)&weightedHistValue,sizeof(weightedHistValue)) ;
        }
    }
    
    binaryFile.close();
}

//------------END OF FUNCTION INPUT

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

cv::Mat loadDescriptors(const string* infile)
{
    
    cv::Mat descriptors, descriptors_float;
    //cout<<"Reading data in"<<endl;
    cv::FileStorage fs(*infile, cv::FileStorage::READ);
    fs["Descriptors"]>>descriptors;
    descriptors.convertTo(descriptors_float, CV_32F);
    //cout<<"Finished reading data"<<endl;
    return descriptors_float;
    
}

flann::Matrix<float> Mat2Float(const cv::Mat* mat)
{
    // allocate and initialize a 2d float array
    //vector<float> V;
    //V.assign((float*)mat->datastart, (float*)mat->dataend);
    //std::cout<<"Converting features into Matrix<float>..."<<std::endl;
    flann::Matrix<float> M((float*)mat->data, mat->rows,mat->cols);
    //std::cout<<"Finished conversion"<<std::endl;
    
    // (use m for something)
    return M;
}



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
    cv::FileStorage fs(*filename, cv::FileStorage::WRITE);
    fs<<"Descriptors"<<m;
}

//Get Image Id
unsigned int get_imageid(const string fname){
    //fname.strmatch();
    int curr_id;
    unsigned foundslash = fname.find_last_of("/");
    //unsigned founddot = fname.find_last_of(".");
    string curr_fname = fname.substr(foundslash+1);
    curr_fname.erase(curr_fname.end()-4,curr_fname.end());
    const char* cfname = curr_fname.c_str();
    curr_id=atoi(cfname);
    //cout<<"Returned id: "<<curr_id<<endl;
    return curr_id;
    
    //return 0;
}

//Extract the SURF Descriptors for each image in a filelist of format vector<string>
cv::Mat computeSURFDescriptors(const std::vector<std::string> Filelist)
{
    int minHessian = 300;
    cv::SurfFeatureDetector detector(minHessian);
    cv::SurfDescriptorExtractor extractor;
    
    cv::Mat img, img_keypoints;
    cv::Mat descriptor;
    
    std::vector<cv::KeyPoint> keypoints_1;
    //std::tuple<std::vector<KeyPoint>, int>;
    
    //FeatureDB surfdesciptors;
    cv::Mat surfdescriptors;
    
    std::cout<<"Number of images to process: "<<Filelist.size()<<std::endl;
    // Compute the SURF Descriptors for each image in the filelist
    for (int i=0; i<Filelist.size();i++) {
        //std::cout<<"Processing Image: "<<Filelist[i]<<std::endl;
        unsigned int img_id = get_imageid(Filelist[i]);
        img = cv::imread(Filelist[i], CV_LOAD_IMAGE_GRAYSCALE);
        if(isLandscape(img))
            img=resizeToFit(img, cvSize(640,480));
        else
            img=resizeToFit(img, cvSize(480, 640));
        
        detector.detect(img, keypoints_1); //Get keypoints
        
        //Compute Descriptors
        extractor.compute(img, keypoints_1, descriptor);
        
        //Store the descriptor
        //surfdesciptors.push_back(descriptor, img_id);
        cv::Mat cols(descriptor.rows, 1,CV_32F, img_id); //Create a column of image ids
        //cols.setTo(img_id);
        
        cv::hconcat(descriptor, cols, descriptor);
        if (i==0) {
            surfdescriptors=descriptor;
        }
        else
            surfdescriptors.push_back(descriptor);
    }
    return surfdescriptors;
}

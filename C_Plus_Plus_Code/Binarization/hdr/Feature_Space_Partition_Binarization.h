//
// Created by tmondal on 19/12/2018.
//

#ifndef CLION_PROJECT_FEATURE_SPACE_PARTITION_BINARIZATION_H
#define CLION_PROJECT_FEATURE_SPACE_PARTITION_BINARIZATION_H


// Libraries
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <unistd.h>
#include <cv.h>
#include <highgui.h>
#include <vector>

using namespace std;
using namespace cv;

// Types
/*struct PointBinar {
    int x;
    int y;
};*/

class Feature_Space_Partition_Binarization {
public:

    Feature_Space_Partition_Binarization();
    static Feature_Space_Partition_Binarization* getInstance();
    virtual ~Feature_Space_Partition_Binarization();


    void auxBinarization(cv::Mat &, cv::Mat &, int, int, float, float, std::string);
    float calcLocalStats(cv::Mat &, cv::Mat &, cv::Mat &, int, int);
    void executeClusteringAndBinarization(IplImage *, IplImage *, cv::Mat);
    cv::Point getRandomPoint(std::vector<cv::Point> &);
    void computeSC(IplImage *, IplImage *, int);
    int convertSCNeighbourhoodToPixel(int, int, int, int);
    float computeMSW(IplImage *, int, int);
    int computeSW(IplImage *);
    void runBinarization(IplImage &, cv::Mat &);

    IplImage MatToIplImage(cv::Mat image1){
        IplImage* image2;
        image2 = cvCreateImage(cvSize(image1.cols,image1.rows),8,3);
        IplImage ipltemp = image1;
        cvCopy(&ipltemp,image2);
        return ipltemp;
    }

    void showImage(cv::Mat& image) {
        cv::namedWindow("Display Image",cv::WINDOW_AUTOSIZE );
        cv::imshow("Display Image", image);
        cvWaitKey(0);
        cvDestroyWindow( "Display Image" );
    }

    void printMatrix(cv::Mat imgMat){
        for (int i = 0; i<imgMat.rows;i++){
            for (int j = 0; j<imgMat.cols;j++){

                cout << (int)imgMat.at<uchar>(i,j) << " " ;
            }
            cout << endl ;
        }
    }
private:
    static Feature_Space_Partition_Binarization* instance;
};


#endif //CLION_PROJECT_FEATURE_SPACE_PARTITION_BINARIZATION_H

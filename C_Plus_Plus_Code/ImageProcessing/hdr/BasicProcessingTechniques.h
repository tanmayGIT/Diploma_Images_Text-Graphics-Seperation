/*
 * BasicProcessingTechniques.h
 *
 *  Created on: Feb 12, 2015
 *      Author: tanmoymondal
 */

#ifndef BASICPROCESSINGTECHNIQUES_H_
#define BASICPROCESSINGTECHNIQUES_H_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/core/core.hpp>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv/cxcore.h"



#include "../../util/hdr/BasicAlgo.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <iterator>
#include <algorithm>
#include <set>
#include <iterator>
#include <numeric>
#include <functional>
#include <list>
#include <stack>          // std::stack
#include <deque>
#include <math.h>
#include <limits>
#include <utility>

#include <cstdio>
#include <cstdlib>

#include "../../Binarization/hdr/PopularDocBinarization.hpp"


using namespace cv;

namespace std {

    struct ComponentInfo{
        int xstart;
        int ystart;
        int height;
        int width;
        int xcenter;
        int ycenter;
        int hullCenterX;
        int hullCenterY;
        long int area;
        vector<cv::Point> outerBlobPts;
        vector<cv::Point> convexHullPts;
        vector<cv::Point> blobFilledPts;
    };
    struct  ConvexHullFeature{
        double majorAxis;
        double minorAxis;
        double orientationEllipseFit;

        double contourPerimeter;
        double contourArea;

        double area;
        double perimeter;
        double compactness;
        double orientationHull;
        double circularity;
        double solidity;
        double convexity;
        double ecentricity;
        double elongation;
    };

    struct BinBucketDivision{
        int binLowerBound;
        int binUpperBound;
    };
    class BasicProcessingTechniques : public BasicAlgo{
    public:
        static BasicProcessingTechniques* getInstance();
        ComponentInfo* connectedComponentLabeling(Mat bin_image, int&);
        int callGetMyMeanValUpdated(vector<int>&, int, std::vector<int>&);
        void getConnectedCompOperationPPPImage(Mat &, vector<ComponentInfo> &, vector<bool> &,  int &, int &);
        BasicProcessingTechniques();
        virtual ~BasicProcessingTechniques();
        Mat Horizontal_RLSATechnique(Mat&,int);

        int getMeanHeightWidthComponentsAdvance(ComponentInfo*, int);      /// This is to calculate Average
        int getMyMeanValUpdated(int *, int, std::vector<int>&);      /// This is to calculate Average
        Mat ComplementImage (Mat);
    private:
        static BasicProcessingTechniques* instance;
        inline Mat GetAverageRowValue(Mat &);



        void cleanupArray(int **hist1, int size1)
        {
            for (int i=0; i<size1; i++)
                 free(hist1[i]);
            free(hist1);
            return;
        }
    };
    
}

#endif /* BASICPROCESSINGTECHNIQUES_H_ */

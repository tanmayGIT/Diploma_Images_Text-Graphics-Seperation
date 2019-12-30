//
//  PopularDocBinarization.hpp
//  wordSpottingNoSegmentation
//
//  Created by Tanmoy on 4/11/16.
//  Copyright © 2016 tanmoy. All rights reserved.
//

#ifndef PopularDocBinarization_hpp
#define PopularDocBinarization_hpp

#include <stdio.h>

#include <unistd.h>
#include <getopt.h>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "../../util/hdr/BasicAlgo.h"
#include "../../ImageProcessing/hdr/BasicProcessingTechniques.h"

using namespace std;
using namespace cv;
#define MAXVAL 256
enum NiblackVersion
{
    NIBLACK=0,
    SAUVOLA,
    WOLFJOLION,
};


#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;

//float sauvola_k("sauvola_k",0.3,"Weighting factor");
//int   sauvola_w("sauvola_w",40,"Local window size. Should always be positive");
//std::string debug_binarize("debug_binarize",0,"output the result of binarization");
struct blockImagProperties{
    int startX;
    int startY;
    int endX;
    int endY;
    Mat croppedImg;
    double meanWindowVal;
    double stdWindowVal;
    double windowThresholdVal;
};
class PopularDocBinarization
{
    
public:
    float k;
    int w;
    int whalf; // Half of window size
    static PopularDocBinarization* getInstance();
    void NiblackSauvolaWolfJolion (Mat, Mat, NiblackVersion, int, int, double, double);
    void callBinarizationFunction(Mat &, Mat &, char);
    void allocateDynamicArray(int, int, long long  int **);
    void releasingDynamicArray(int, long long  int **);
    void binarizeSafait(Mat &, Mat &);
    void binarizeDynamicWindows(Mat &, Mat &);
    void divideImageIntoSubImages(Mat &, int, int, vector<blockImagProperties>&, double&, double&);
    void getBinarizedImageSubSubParts(Mat&, Mat&, int, int, int, int, double, double, double, float );

    void NiblackSauvolaWolfJolionFeature (Mat&, Mat&, Mat&, NiblackVersion, int, int, double, double);
    void callBinarizationFeature(Mat& , Mat& , Mat& , int , int , char );


    void CalculateNiblackFeatures (Mat&, Mat&, int, int);
    void CalculateSavoulaFeatures (Mat&, Mat&, int, int, float);
    void BolanSUFeatures (Mat, Mat);
    void set(const char *,double );
    void BolanSUFeatures (Mat&, Mat&, int, int);
    void HoweFeatures (Mat&, Mat&, int, int);
    void CalculateFeaturesForBinarization(Mat &, Mat &);
    void Calculate_Mean_StandardDeviation_Features(Mat&, Mat&, Mat&, int, int);
    void printFloatMatrix(cv::Mat);
    void CreateBackGroundImage(Mat&, Mat&);
    void OrigImageNormalization_by_BackGdImage(Mat&, Mat&, Mat&);
    inline void DoBackGroundImageOperation(Mat&, Mat&, Mat&, int, int);
    int CalculateStrokeWidth(Mat&,  Mat&);
    void LineWiseDivideImageIntoSubImages(Mat &, int, int , int, int, vector<blockImagProperties>&);

    void Do_Another_Image_Normalization(Mat&, Mat& , Mat&);
    void Gradient_Image_Calculation(Mat&, Mat&);
    void SSP_Extraction( Mat&, Mat&);

    int reflect(int, int);
    void LogIntensityPercentileFeatures (Mat&, Mat&, int, int);
    void RelativeDarknessIndexFeatures (Mat&, Mat&, int);

    void CalculateHorizontal_X_Border_Image(Mat&, int&, int& );
    void CalculateVertical_Y_Border_Image(Mat&, int&, int& );

    void SimpleDynamicWindowBased_Binarization(Mat&, Mat&, Mat&);
    void GetSomeStatistics_From_FullImage(Mat&, float& , float& , double& , double&  );
    float euclideanDist(Point& p, Point& q) {
        Point diff = p - q;
        return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
    }

    static PopularDocBinarization *instance()
    {
        if (!s_instance)
            s_instance = new PopularDocBinarization;
        return s_instance;
    }
    PopularDocBinarization(){
        w = 40;
        k = 0.3;
        
    }





private:
    static PopularDocBinarization *s_instance;
    double calcLocalStats (Mat &, Mat &, Mat &, int, int);
    
};

#endif /* PopularDocBinarization_hpp */

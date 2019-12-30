//
//  ImageFiltering.hpp
//  wordSpottingNoSegmentation
//
//  Created by Tanmoy on 3/26/16.
//  Copyright © 2016 tanmoy. All rights reserved.
//

#ifndef ImageFiltering_hpp
#define ImageFiltering_hpp


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <iomanip>

#include <stdlib.h>
#include <algorithm>
#include <set>
#include <iterator>
#include <numeric>
#include <functional>
#include <list>
#include <math.h>
#include <limits>

#include <cstdio>
#include <cstdlib>

#include "../../ImageProcessing/hdr/BasicProcessingTechniques.h"
// #include "../../ImageFiltering/hdr/GaussianProbFeatureStructures.hpp"
//#include "omp.h"

#define PI 3.14159265358979323846

using namespace std;
using namespace cv;

static const char *GAUSSIAN_FILTER_NAME = "gaussian";
const int GAUSSIAN_RADIUS = 3;
const int MOTION_X = 40;
const int MOTION_Y = 5;


struct FilteringMeanHeightWidth{
    int mean_height;
    int mean_width;
};

class ImageFiltering{
public:
 
    typedef struct {
        double value[3];
        double weight[3];
    } Weights;
    
    static ImageFiltering* getInstance();
    Mat simpleGaussianFilter(Mat, float );
    void createFilter(double gKernel[][5]);
    
    ImageFiltering();
    virtual ~ImageFiltering();

    IplImage* gaussianBlur(IplImage* , double );
    //IplImage* lineAveragingFilter(IplImage* , float, float, float, double );
    IplImage* gaussianBlurParallel(IplImage* , double );
    Mat& removeSmallNoisesDots(Mat&);
    IplImage* motionBlur(IplImage* , int ,  int);
    IplImage* motionBlurParallel(IplImage* , int ,  int );
    void meanFilter(Mat &image);

    bool isParallel(const char * parallel);
    FilteringMeanHeightWidth CV_EXPORTS applyFloodFill( Mat&, Mat& );
    IplImage* doImfill(IplImage* );
    
    int myMin(int a, int b) {
        return a < b ? a : b;
    }
    
    int myMax(int a, int b) {
        return a > b ? a : b;
    }
    template<typename T>
    std::vector<T> conv_valid(std::vector<T>&, std::vector<T>&);
    std::vector <float> ComputeGaussianKernel(const int, const float);
private:
    static ImageFiltering* instance;
};
#endif /* ImageFiltering_hpp */

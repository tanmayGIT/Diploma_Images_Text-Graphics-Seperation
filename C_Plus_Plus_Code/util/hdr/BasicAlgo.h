/*
 * BasicAlgo.h
 *
 *  Created on: Feb 18, 2015
 *      Author: tanmoymondal
 */

#ifndef BASICALGO_H_
#define BASICALGO_H_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <opencv2/features2d/features2d.hpp>

#include "opencv/cv.h"
#include "opencv/cxcore.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <algorithm>
#include <set>
#include <iterator>
#include <string.h>
#include <math.h>
#include <limits>
#include <stdexcept>

using namespace cv;
using namespace std;

class BasicAlgo {
public:
    BasicAlgo();

    static BasicAlgo *getInstance();

    virtual ~BasicAlgo();

    void showImage(cv::Mat &);
    void writeImage(cv::Mat &);
    void writeImageGivenPath(cv::Mat &, string);

    int* quicksort (int*, int, int);
    Mat convertToMat(unsigned char *buffer, int height, int width) {
        Mat tmp(height, width, CV_8UC1);
        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                int value = (int) buffer[x * width + y];
                tmp.at<uchar>(x,y) = value;
            }
        }
        return tmp;
    }

    std::vector<float> uniqueValuesInMat(const cv::Mat& rawData, bool sort = false)
    {
        Mat input;
        rawData.convertTo(input, CV_32F);
        if (input.channels() > 1 || input.type() != CV_32F)
        {
            std::cerr << "unique !!! Only works with CV_32F 1-channel Mat" << std::endl;
            return std::vector<float>();
        }

        std::vector<float> out;
        for (int y = 0; y < input.rows; ++y)
        {
            const float* row_ptr = input.ptr<float>(y);
            for (int x = 0; x < input.cols; ++x)
            {
                float value = row_ptr[x];

                if ( std::find(out.begin(), out.end(), value) == out.end() )
                    out.push_back(value);
            }
        }

        if (sort)
            std::sort(out.begin(), out.end());

        return out;
    }

private:
    static BasicAlgo *instance;
};


#endif /* BASICALGO_H_ */

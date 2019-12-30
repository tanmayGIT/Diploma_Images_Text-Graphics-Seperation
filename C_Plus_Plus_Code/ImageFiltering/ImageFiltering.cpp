//
//  ImageFiltering.cpp
//  wordSpottingNoSegmentation
//
//  Created by Tanmoy on 3/26/16.
//  Copyright © 2016 tanmoy. All rights reserved.
//

#include "ImageFiltering.hpp"

ImageFiltering* ImageFiltering::instance = 0;

ImageFiltering* ImageFiltering::getInstance() {
    if (!instance)
        instance = new ImageFiltering();
    
    return instance;
}

ImageFiltering::ImageFiltering() {
}

ImageFiltering::~ImageFiltering() {
}

template<typename T>
std::vector<T>
ImageFiltering::conv_valid(std::vector<T> &f, std::vector<T> &g) {
    int const nf = f.size();
    int const ng = g.size();
    int const n  = nf + ng - 1;
    std::vector<T> out(n, T());
    for(auto i(0); i < n; ++i) {
        int const jmn = (i >= ng - 1)? i - (ng - 1) : 0;
        int const jmx = (i <  nf - 1)? i            : nf - 1;
        for(auto j(jmn); j <= jmx; ++j) {
            out[i] += (f[j] * g[i - j]);
        }
    }
    return out;
}
std::vector <float> ImageFiltering::ComputeGaussianKernel(const int inRadius, const float inWeight)
{
    int mem_amount = (inRadius*2)+1;
    float sigma = (inRadius-1)/3;
    std::vector <float> gaussian_kernel(mem_amount);
    
    float twoRadiusSquaredRecip = 1.0 / (2.0 * sigma * sigma);
    float sqrtTwoPiTimesRadiusRecip = 1.0 / (sqrt(2.0 * PI) * sigma);
    float radiusModifier = inWeight;
    
    // Create Gaussian Kernel
    int r = -inRadius;
    float sum = 0.0f;
    for (int i = 0; i < mem_amount; i++)
    {
        float x = r * radiusModifier;
        x *= x;
        float v = sqrtTwoPiTimesRadiusRecip * exp(-x * twoRadiusSquaredRecip);
        gaussian_kernel[i] = v;
        
        sum+=v;
        r++;
    }
    
    // Normalize distribution
    float div = sum;
    for (int i = 0; i < mem_amount; i++)
        gaussian_kernel[i] /= div;
    this->conv_valid(gaussian_kernel, gaussian_kernel);
    return gaussian_kernel;
}


FilteringMeanHeightWidth CV_EXPORTS ImageFiltering::applyFloodFill( Mat& _src, Mat& dst)
{
    CV_Assert(_src.type() == CV_8UC1);

    vector<vector<cv::Point>> contours;
    vector<Vec4i> hierarchy;

    findContours(_src,contours,hierarchy,CV_RETR_CCOMP,CV_CHAIN_APPROX_SIMPLE,cv::Point(0,0));
    CvScalar color = cvScalar(255);

    int numComponent = contours.size();
    int* heightKeeper = NULL;
    heightKeeper = new int[numComponent];
    int* widthKeeper = NULL;
    widthKeeper = new int[numComponent];

    for(int i = 0;i < contours.size(); i++)
    {
        drawContours(dst,contours,i,color,-1,8,hierarchy,0,cv::Point());
        heightKeeper[i] = (boundingRect(contours[i]).height);
        widthKeeper[i] = (boundingRect(contours[i]).width);
    }
    std::vector<int>assembleGoodEleRefined;
    FilteringMeanHeightWidth keepHeightWidth;
    int avgCompHeight = BasicProcessingTechniques::getInstance()->getMyMeanValUpdated(heightKeeper,numComponent, assembleGoodEleRefined);
    int avgCompWidth = BasicProcessingTechniques::getInstance()->getMyMeanValUpdated(widthKeeper,numComponent, assembleGoodEleRefined);
    keepHeightWidth.mean_height = avgCompHeight;
    keepHeightWidth.mean_width = avgCompWidth;
    return keepHeightWidth;
}

IplImage* ImageFiltering::doImfill(IplImage* src)
{
    CvScalar white = CV_RGB( 255, 255, 255 );

    IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3);
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* contour = 0;

    cvFindContours(src, storage, &contour, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );
    cvZero( dst );

    for( ; contour != 0; contour = contour->h_next )
    {
        cvDrawContours( dst, contour, white, white, 0, CV_FILLED);
    }

    IplImage* bin_imgFilled = cvCreateImage(cvGetSize(src), 8, 1);
    cvInRangeS(dst, white, white, bin_imgFilled);

    return bin_imgFilled;
}

Mat& ImageFiltering::removeSmallNoisesDots(Mat& noisyImage){
    
    /* Taken from the following link :
     docs.opencv.org/2.4/doc/tutorials/imgproc/opening_closing_hats/opening_closing_hats.html
     (it is assumed that the objects are bright on a dark foreground)
     */
    // Define the structuring elements
    Mat se1 = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat se2 = getStructuringElement(MORPH_RECT, Size(2, 2));
    
    // Perform closing then opening
    static Mat mask;
    morphologyEx(noisyImage, mask, MORPH_CLOSE, se1);
    morphologyEx(mask, mask, MORPH_OPEN, se2);
    
    // Filter the output
//    Mat out = noisyImage.clone();
//    out.setTo(Scalar(0), mask == 0); // set all the pixels of out to 0 which has 0 value at the same indexes in mask matrix
    return mask;
}

IplImage* ImageFiltering::gaussianBlur(IplImage* image, double r) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
    printf("h=%d, w=%d", h, w);
    double rs = ceil(r * 2.57);     // significant radius
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = i-rs; iy<i+rs+1; iy++) {
                for (int ix = j - rs; ix < j + rs + 1; ix++) {
                    int x = myMin(w - 1, myMax(0, ix));
                    int y = myMin(h - 1, myMax(0, iy));
                    double dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
                    double wght = exp(-dsq / (2 * r * r)) / (PI * 2 * r * r);
                    CvScalar channels = cvGet2D(image, y, x);
                    
                    // calculate the value for each channel
                    for (int c = 0; c < 3; c++) {
                        weights.value[c] += channels.val[c] * wght;
                        weights.weight[c] += wght;
                    }
                }
            }
            
            // set the value for each channel in the resulting image.
            //            printf("i=%d, j=%d, r=%f, g=%f, b=%f\n", i, j, weights.value[0], weights.value[1], weights.value[2]);
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = round(weights.value[c] / weights.weight[c]);
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}

//IplImage* ImageFiltering::lineAveragingFilter(IplImage* image, float lStart, float lEnd, float thetaRange, double r) {
//    IplImage* result = cvCloneImage(image);
//    int h = image->height;
//    int w = image->width;
//    IplImage* gaussImage = this->gaussianBlur(image, r);
//    
//    for(int i=0; i<h; i++) {
//        for (int j = 0; j < w; j++) {
//            Weights weights;
//            for(int iy = i-sigmaY; iy<i+sigmaY+1; iy++) {
//                for (int ix = j - sigmaX; ix < j + sigmaX + 1; ix++) {
//                    int x = myMin(w - 1, myMax(0, ix));
//                    int y = myMin(h - 1, myMax(0, iy));
//                    float part_1 = ((((ix - j) * cos(theta) + (iy - i) * sin(theta))^2)/sigmaX^2);
//                    float part_2 = (((-(iy - i) * sin(theta) + (iy - i) * cos(theta))^2)/sigmaY^2);
//                    double dsq = (part_1 + part_2)/2;
//                    double wght = exp(-dsq / (PI * 2 * sigmaX * sigmaY);
//                    CvScalar channels = cvGet2D(image, y, x);
//                    
//                    // calculate the value for each channel
//                    for (int c = 0; c < 3; c++) {
//                        weights.value[c] += channels.val[c] * wght;
//                        weights.weight[c] += wght;
//                    }
//                }
//            }
//            
//            // set the value for each channel in the resulting image.
//            //            printf("i=%d, j=%d, r=%f, g=%f, b=%f\n", i, j, weights.value[0], weights.value[1], weights.value[2]);
//            CvScalar resultingChannels = cvGet2D(result, i, j);
//            for(int c=0; c < 3; c++) {
//                resultingChannels.val[c] = round(weights.value[c] / weights.weight[c]);
//                weights.value[c] = 0.0;
//                weights.weight[c] = 0.0;
//            }
//            cvSet2D(result, i, j, resultingChannels);
//        }
//    }
//    return result;
//}

IplImage* ImageFiltering::gaussianBlurParallel(IplImage* image, double r) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
    
    double rs = ceil(r * 2.57);     // significant radius
#pragma omp parallel for schedule(guided)
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = i-rs; iy<i+rs+1; iy++) {
                for (int ix = j - rs; ix < j + rs + 1; ix++) {
                    int x = myMin(w - 1, myMax(0, ix));
                    int y = myMin(h - 1, myMax(0, iy));
                    double dsq = (ix - j) * (ix - j) + (iy - i) * (iy - i);
                    double wght = exp(-dsq / (2 * r * r)) / (PI * 2 * r * r);
                    CvScalar channels = cvGet2D(image, y, x);
                    
                    // calculate the value for each channel
                    for (int c = 0; c < 3; c++) {
                        weights.value[c] += channels.val[c] * wght;
                        weights.weight[c] += wght;
                    }
                }
            }
            
            // set the value for each channel in the resulting image.
            // printf("i=%d, j=%d, r=%f, g=%f, b=%f\n", i, j, weights.value[0], weights.value[1], weights.value[2]);
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = round(weights.value[c] / weights.weight[c]);
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}






void ImageFiltering::meanFilter(Mat &image) {
   // static Mat filtered = Mat::zeros(image.rows, image.cols, CV_8U);
    /*
     For each pixel in noisy image starting from 1 to N-1 on each direction,
     get the pixel values of 3×3 area (i-1->i+1,j-1->j+1) and store them in the kernel array.
     */
    for (int i = 1; i < (image.rows - 1); i++) {
        for (int j = 1; j < (image.cols - 1); j++) {
            int kernalIndex = 0;
            int kernalAdder  = 0;
            for (int a = -1; a <= 1; a++) {
                for (int b = -1; b <= 1; b++) {
                    kernalAdder = kernalAdder + image.at<uchar>(i + a, j + b);
                    kernalIndex++;
                }
            }
            image.at<uchar>(i, j) = (int) kernalAdder/kernalIndex;
        }
    }
}

IplImage* ImageFiltering::motionBlur(IplImage* image, int deltaX,  int deltaY) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = myMin(i + deltaY, h -1) ; iy > i; iy--) {
                CvScalar channels = cvGet2D(image, iy, j);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c] * 2;
                    weights.weight[c] += 2;
                }
            }
            
            for(int ix = myMin(j + deltaX, w-1); ix > j; ix--) {
                CvScalar channels = cvGet2D(image, i, ix);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c];
                    weights.weight[c] += 1;
                }
            }
            
            // set the value for each channel in the resulting image.
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = weights.value[c] / weights.weight[c];
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}

IplImage* ImageFiltering::motionBlurParallel(IplImage* image, int deltaX,  int deltaY) {
    IplImage* result = cvCloneImage(image);
    int h = image->height;
    int w = image->width;
#pragma omp parallel for schedule(guided)
    for(int i=0; i<h; i++) {
        for (int j = 0; j < w; j++) {
            Weights weights;
            for(int iy = myMin(i + deltaY, h -1) ; iy > i; iy--) {
                CvScalar channels = cvGet2D(image, iy, j);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c] * 2;
                    weights.weight[c] += 2;
                }
            }
            
            for(int ix = myMin(j + deltaX, w-1); ix > j; ix--) {
                CvScalar channels = cvGet2D(image, i, ix);
                // calculate the value for each channel
                for (int c = 0; c < 3; c++) {
                    weights.value[c] += channels.val[c];
                    weights.weight[c] += 1;
                }
            }
            
            // set the value for each channel in the resulting image.
            CvScalar resultingChannels = cvGet2D(result, i, j);
            for(int c=0; c < 3; c++) {
                resultingChannels.val[c] = weights.value[c] / weights.weight[c];
                weights.value[c] = 0.0;
                weights.weight[c] = 0.0;
            }
            cvSet2D(result, i, j, resultingChannels);
        }
    }
    return result;
}

bool isParallel(const char * parallel) {
    return strcmp(parallel, "true") == 0;
}

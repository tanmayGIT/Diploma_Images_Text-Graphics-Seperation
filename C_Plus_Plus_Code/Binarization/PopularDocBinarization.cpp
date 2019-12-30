//
//  PopularDocBinarization.cpp
//  wordSpottingNoSegmentation
//
//  Created by Tanmoy on 4/11/16.
//  Copyright © 2016 tanmoy. All rights reserved.
//

#include "hdr/PopularDocBinarization.hpp"



// *************************************************************
// glide a window across the image and
// create two maps: mean and standard deviation.
// *************************************************************
PopularDocBinarization *PopularDocBinarization::s_instance = 0;

PopularDocBinarization* PopularDocBinarization::getInstance() {
    if (!s_instance)
        s_instance = new PopularDocBinarization();
    return s_instance;
}

double PopularDocBinarization::calcLocalStats (Mat &im, Mat &map_m, Mat &map_s, int winx, int winy) {
    Mat im_sum, im_sum_sq;
    cv::integral(im,im_sum,im_sum_sq,CV_64F);
    
    double m,s,max_s,sum,sum_sq;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    double winarea = winx*winy;
    
    max_s = 0;
    for	(int j = y_firstth ; j<=y_lastth; j++){
        sum = sum_sq = 0;
        
        sum = im_sum.at<double>(j-wyh+winy,winx) - im_sum.at<double>(j-wyh,winx) - im_sum.at<double>(j-wyh+winy,0) + im_sum.at<double>(j-wyh,0);
        sum_sq = im_sum_sq.at<double>(j-wyh+winy,winx) - im_sum_sq.at<double>(j-wyh,winx) - im_sum_sq.at<double>(j-wyh+winy,0) + im_sum_sq.at<double>(j-wyh,0);
        
        m  = sum / winarea;
        s  = sqrt ((sum_sq - m*sum)/winarea);
        if (s > max_s) max_s = s;
        
        map_m.fset(x_firstth, j, m);
        map_s.fset(x_firstth, j, s);
        
        // Shift the window, add and remove	new/old values to the histogram
        for	(int i=1 ; i <= im.cols-winx; i++) {
            
            // Remove the left old column and add the right new column
            sum -= im_sum.at<double>(j-wyh+winy,i) - im_sum.at<double>(j-wyh,i) - im_sum.at<double>(j-wyh+winy,i-1) + im_sum.at<double>(j-wyh,i-1);
            sum += im_sum.at<double>(j-wyh+winy,i+winx) - im_sum.at<double>(j-wyh,i+winx) - im_sum.at<double>(j-wyh+winy,i+winx-1) + im_sum.at<double>(j-wyh,i+winx-1);
            
            sum_sq -= im_sum_sq.at<double>(j-wyh+winy,i) - im_sum_sq.at<double>(j-wyh,i) - im_sum_sq.at<double>(j-wyh+winy,i-1) + im_sum_sq.at<double>(j-wyh,i-1);
            sum_sq += im_sum_sq.at<double>(j-wyh+winy,i+winx) - im_sum_sq.at<double>(j-wyh,i+winx) - im_sum_sq.at<double>(j-wyh+winy,i+winx-1) + im_sum_sq.at<double>(j-wyh,i+winx-1);
            
            m  = sum / winarea;
            s  = sqrt ((sum_sq - m*sum)/winarea);
            if (s > max_s) max_s = s;
            
            map_m.fset(i+wxh, j, m);
            map_s.fset(i+wxh, j, s);
        }
    }
    
    return max_s;
}

/**********************************************************
 * The binarization routine
 **********************************************************/


void PopularDocBinarization::NiblackSauvolaWolfJolion (Mat im, Mat output, NiblackVersion version, int winx, int winy, double k, double dR) {
    
    double m, s, max_s;
    double th=0;
    double min_I, max_I;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int x_lastth = im.cols-wxh-1;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    
    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat map_s = Mat::zeros (im.rows, im.cols, CV_32F);
    max_s = calcLocalStats (im, map_m, map_s, winx, winy);
    
    minMaxLoc(im, &min_I, &max_I);
    
    Mat thsurf (im.rows, im.cols, CV_32F);
    
    // Create the threshold surface, including border processing
    // ----------------------------------------------------
    
    for	(int j = y_firstth ; j<=y_lastth; j++) {
        
        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
        for	(int i=0 ; i <= im.cols-winx; i++) {
            
            m  = map_m.fget(i+wxh, j);
            s  = map_s.fget(i+wxh, j);
            
            // Calculate the threshold
            switch (version) {
                    
                case NIBLACK:
                    th = m + k*s;
                    break;
                    
                case SAUVOLA:
                    th = m * (1 + k*(s/dR-1));
                    break;
                    
                case WOLFJOLION:
                    th = m + k * (s/max_s-1) * (m-min_I);
                    break;
                    
                default:
                    cerr << "Unknown threshold type in ImageThresholder::surfaceNiblackImproved()\n";
                    exit (1);
            }
            
            thsurf.fset(i+wxh,j,th);
            
            if (i==0) {
                // LEFT BORDER
                for (int i=0; i<=x_firstth; ++i)
                    thsurf.fset(i,j,th);
                
                // LEFT-UPPER CORNER
                if (j==y_firstth)
                    for (int u=0; u<y_firstth; ++u)
                        for (int i=0; i<=x_firstth; ++i)
                            thsurf.fset(i,u,th);
                
                // LEFT-LOWER CORNER
                if (j==y_lastth)
                    for (int u=y_lastth+1; u<im.rows; ++u)
                        for (int i=0; i<=x_firstth; ++i)
                            thsurf.fset(i,u,th);
            }
            
            // UPPER BORDER
            if (j==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    thsurf.fset(i+wxh,u,th);
            
            // LOWER BORDER
            if (j==y_lastth)
                for (int u=y_lastth+1; u<im.rows; ++u)
                    thsurf.fset(i+wxh,u,th);
        }
        
        // RIGHT BORDER
        for (int i=x_lastth; i<im.cols; ++i)
            thsurf.fset(i,j,th);
        
        // RIGHT-UPPER CORNER
        if (j==y_firstth)
            for (int u=0; u<y_firstth; ++u)
                for (int i=x_lastth; i<im.cols; ++i)
                    thsurf.fset(i,u,th);
        
        // RIGHT-LOWER CORNER
        if (j==y_lastth)
            for (int u=y_lastth+1; u<im.rows; ++u)
                for (int i=x_lastth; i<im.cols; ++i)
                    thsurf.fset(i,u,th);
    }
    
/*    namedWindow("Binarization Output", WINDOW_AUTOSIZE);
    imshow("Binarized", thsurf);
    
    cerr << "surface created" << endl;*/
    
    
    for	(int y=0; y<im.rows; ++y)
        for	(int x=0; x<im.cols; ++x)
        {
            if (im.uget(x,y) >= thsurf.fget(x,y))
            {
                output.uset(x,y,255);
            }
            else
            {
                output.uset(x,y,0);
            }
        }
}



void PopularDocBinarization::callBinarizationFunction(Mat& input, Mat& output, char version){
    int winx=0, winy=0;
    float optK=0.5;
    NiblackVersion versionCode;
    
    // Determine the method
    switch (version)
    {
        case 'n':
            versionCode = NIBLACK;
            cerr << "Niblack (1986)\n";
            break;
            
        case 's':
            versionCode = SAUVOLA;
            cerr << "Sauvola et al. (1997)\n";
            break;
            
        case 'w':
            versionCode = WOLFJOLION;
            cerr << "Wolf and Jolion (2001)\n";
            break;
            
        default:
            cerr  << "\nInvalid version: '" << version << "'!";
    }
    
    // Treat the window size
    if (winx==0||winy==0) {
        cerr << "Input size: " << input.cols << "x" << input.rows << endl;
        winy = (int) (2.0 * input.rows-1)/3;
        winx = (int) input.cols-1 < winy ? input.cols-1 : winy;
        // if the window is too big, than we asume that the image
        // is not a single text box, but a document page: set
        // the window size to a fixed constant.
        if (winx > 100)
            winx = winy =  40;
        cerr << "Setting window size to [" << winx
        << "," << winy << "].\n";
    }
    
    // Threshold
    NiblackSauvolaWolfJolion (input, output, versionCode, winx, winy, optK, 128);
}




void PopularDocBinarization::NiblackSauvolaWolfJolionFeature (Mat& im, Mat& output, Mat& SSP_Imag, NiblackVersion version, int winx, int winy, double k, double dR) {

    double m, s, max_s;
    double th=0;
    double min_I, max_I;
    int wxh	= winx/2;
    int wyh	= winy/2;
    int x_firstth= wxh;
    int x_lastth = im.cols-wxh-1;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;


    Scalar value;
    RNG rng(12345);
    Mat imCopy = im; // keeping the original image before padding
    Mat SSP_Imag_Copy = SSP_Imag; // keeping the original SSP_Imag before padding

    value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    copyMakeBorder(im, im, (winy / 2), (winy / 2), (winx / 2), (winx / 2), BORDER_REPLICATE, value);
    copyMakeBorder(SSP_Imag, SSP_Imag, (winy / 2), (winy / 2), (winx / 2), (winx / 2), BORDER_REPLICATE, value);


    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat map_s = Mat::zeros (im.rows, im.cols, CV_32F);
    max_s = calcLocalStats (im, map_m, map_s, winx, winy);

    minMaxLoc(im, &min_I, &max_I);

    Mat thsurf  = Mat::zeros(im.rows, im.cols, CV_32F);

    // Create the threshold surface, including border processing
    // ----------------------------------------------------
    for	(int j = y_firstth ; j<=y_lastth; j++) {

        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
        for	(int i=0 ; i <= im.cols-winx; i++) {
            if (SSP_Imag.at<uchar>(j, i) == 255) {

                m = map_m.fget(i + wxh, j);
                s = map_s.fget(i + wxh, j);

                // Calculate the threshold
                switch (version) {

                    case NIBLACK:
                        th = m + k * s;
                        break;

                    case SAUVOLA:
                        th = m * (1 + k * (s / dR - 1));
                        break;

                    case WOLFJOLION:
                        th = m + k * (s / max_s - 1) * (m - min_I);
                        break;

                    default:
                        cerr << "Unknown threshold type in ImageThresholder::surfaceNiblackImproved()\n";
                        exit(1);
                }

                thsurf.fset(i + wxh, j, th);

                if (i == 0) {
                    // LEFT BORDER
                    for (int i = 0; i <= x_firstth; ++i)
                        thsurf.fset(i, j, th);

                    // LEFT-UPPER CORNER
                    if (j == y_firstth)
                        for (int u = 0; u < y_firstth; ++u)
                            for (int i = 0; i <= x_firstth; ++i)
                                thsurf.fset(i, u, th);

                    // LEFT-LOWER CORNER
                    if (j == y_lastth)
                        for (int u = y_lastth + 1; u < im.rows; ++u)
                            for (int i = 0; i <= x_firstth; ++i)
                                thsurf.fset(i, u, th);
                }

                // UPPER BORDER
                if (j == y_firstth)
                    for (int u = 0; u < y_firstth; ++u)
                        thsurf.fset(i + wxh, u, th);

                // LOWER BORDER
                if (j == y_lastth)
                    for (int u = y_lastth + 1; u < im.rows; ++u)
                        thsurf.fset(i + wxh, u, th);
            }
        }

            // RIGHT BORDER
            for (int i = x_lastth; i < im.cols; ++i)
                thsurf.fset(i, j, th);

            // RIGHT-UPPER CORNER
            if (j == y_firstth)
                for (int u = 0; u < y_firstth; ++u)
                    for (int i = x_lastth; i < im.cols; ++i)
                        thsurf.fset(i, u, th);

            // RIGHT-LOWER CORNER
            if (j == y_lastth)
                for (int u = y_lastth + 1; u < im.rows; ++u)
                    for (int i = x_lastth; i < im.cols; ++i)
                        thsurf.fset(i, u, th);
    }
    output = thsurf.clone();
    // BasicAlgo::getInstance()->showImage(thsurf);
    // std::vector<float> getAllUniqueVals = BasicAlgo::getInstance()->uniqueValuesInMat(thsurf,true);
    thsurf.release();

    SSP_Imag = SSP_Imag_Copy;
    SSP_Imag_Copy.release();

    im = imCopy;
    imCopy.release();

    Mat outputCropped = output(Rect((winx/2), (winy/2), im.cols, im.rows));
    output = outputCropped;
    outputCropped.release();

 //   BasicAlgo::getInstance()->showImage(output);
}



void PopularDocBinarization::callBinarizationFeature(Mat& input, Mat& output, Mat& SSP_Imag, int winx, int winy, char version){

    float optK=0.5;
    NiblackVersion versionCode;

    // Determine the method
    switch (version)
    {
        case 'n':
            versionCode = NIBLACK;
            cerr << "Niblack (1986)\n";
            break;

        case 's':
            versionCode = SAUVOLA;
            cerr << "Sauvola et al. (1997)\n";
            break;

        case 'w':
            versionCode = WOLFJOLION;
            cerr << "Wolf and Jolion (2001)\n";
            break;

        default:
            cerr  << "\nInvalid version: '" << version << "'!";
    }

    // Threshold
    NiblackSauvolaWolfJolionFeature (input, output, SSP_Imag, versionCode, winx, winy, optK, 128);
}


// Allocate a dynalic 2D array of type long long int and initialize it with 0
void PopularDocBinarization::allocateDynamicArray(int nRows, int nCols, long long  int **arrayName){
    arrayName = new long long int*[nRows];
    for(int i = 0;i < nRows; i++)
    {
        arrayName[i] = new long long int[nCols];
    }
    // for initializing the array with zeros
    for (int i = 0; i < nRows*nCols; i++)
        *((int*)arrayName + i) = 0;
    
}

void PopularDocBinarization::releasingDynamicArray(int nRows, long long  int **arrayName){
    for(int i = 0; i < nRows; i++)
        delete []arrayName[i];
    delete [] arrayName;
    
}


void PopularDocBinarization::LineWiseDivideImageIntoSubImages(Mat &gray_image, int windowHeight, int windowWidth,int startLineRow, int endLineRow,
                                                      vector<blockImagProperties>& keepAllBlockImag) {
    //  int heightQuotient = gray_image.rows / windowHeight;
    int heightRemainder = ((endLineRow - startLineRow)) % windowHeight;

    int endingHeight = 0;
    int endingWidth = 0;
    if (heightRemainder > 0) {
        endingHeight = endLineRow - (windowHeight + heightRemainder);
    } else {
        endingHeight = endLineRow - (windowHeight + 0);
    }


    blockImagProperties prepareBlocks;
    cv::Rect getROI;
    int iRow = 0, jCol = 0;

    // Mat ReconstructImage = Mat::ones(gray_image.size(), gray_image.type());

    for (iRow = startLineRow; iRow < endingHeight; iRow = iRow + windowHeight) {
        getROI.x = jCol;
        getROI.y = iRow;
        getROI.width = windowWidth; // put the whole width of the image
        getROI.height = windowHeight;

        Mat croppedImg = gray_image(getROI);
        prepareBlocks.startX = getROI.x;
        prepareBlocks.startY = getROI.y;
        prepareBlocks.endX = getROI.width;
        prepareBlocks.endY = getROI.height;
        prepareBlocks.croppedImg = croppedImg;
        keepAllBlockImag.push_back(prepareBlocks);

        // croppedImg.copyTo(ReconstructImage(getROI));
    }
    // BasicAlgo::getInstance()->showImage(ReconstructImage);

    // For the end row blocks
    getROI.x = jCol;
    getROI.y = iRow; // good
    getROI.width = windowWidth;
    getROI.height = (endLineRow - iRow); // good because it is taking the remaining
    Mat croppedImg = gray_image(getROI);

    prepareBlocks.startX = getROI.x;
    prepareBlocks.startY = getROI.y;
    prepareBlocks.endX = getROI.width;
    prepareBlocks.endY = getROI.height;
    prepareBlocks.croppedImg = croppedImg;

    keepAllBlockImag.push_back(prepareBlocks);

    // croppedImg.copyTo(ReconstructImage(getROI));
    // BasicAlgo::getInstance()->showImage(ReconstructImage);
}




void PopularDocBinarization::divideImageIntoSubImages(Mat &gray_image, int windowHeight, int windowWidth,
                                                      vector<blockImagProperties>& keepAllBlockImag,
                                                      double& minStandardDev, double& maxStandardDev) {
   //  int heightQuotient = gray_image.rows / windowHeight;
    int heightRemainder = gray_image.rows % windowHeight;

    int endingHeight = 0;
    int endingWidth = 0;
    if (heightRemainder > 0) {
        endingHeight = gray_image.rows - (windowHeight + heightRemainder);
    } else {
        endingHeight = gray_image.rows - (windowHeight + 0);
    }

   //  int widthQuotient = gray_image.cols / windowWidth;
    int widthRemainder = gray_image.cols % windowWidth;

    if (widthRemainder > 0) {
        endingWidth = gray_image.cols - (windowWidth + widthRemainder);
    } else {
        endingWidth = gray_image.cols - (windowHeight + 0);
    }
    blockImagProperties prepareBlocks;

    cv::Rect getROI;
    int iRow = 0, jCol = 0;

    for (iRow = 0; iRow < endingHeight; iRow = iRow + windowHeight) {
        for (jCol = 0; jCol < endingWidth; jCol = jCol + windowWidth) {
            getROI.x = jCol;
            getROI.y = iRow;
            getROI.width = windowWidth;
            getROI.height = windowHeight;

            Mat croppedImg = gray_image(getROI);
            prepareBlocks.startX = getROI.x;
            prepareBlocks.startY = getROI.y;
            prepareBlocks.endX = getROI.x + getROI.width;
            prepareBlocks.endY = getROI.y + getROI.height;
            prepareBlocks.croppedImg = croppedImg;

/*            cv::Rect rect(getROI.x, getROI.y, getROI.width, getROI.height);
            cv::rectangle(gray_image, rect, cv::Scalar(0, 255, 0));*/


            Scalar meanWindowVal;
            Scalar stdWindowVal;
            meanStdDev(croppedImg, meanWindowVal, stdWindowVal);
            double pickMeanVal = meanWindowVal.val[0];
            double pickStdVal = stdWindowVal.val[0];

            prepareBlocks.stdWindowVal = pickStdVal;
            prepareBlocks.meanWindowVal = pickMeanVal;

            keepAllBlockImag.push_back(prepareBlocks);

            if (pickStdVal < minStandardDev)  // getting the minimum standard deviation
                minStandardDev = pickStdVal;

            if (pickStdVal > maxStandardDev)  // getting the maximum standard Deviation
                maxStandardDev = pickStdVal;
        }
    }
 //   BasicAlgo::getInstance()->showImage(gray_image);

    // For the end row
    for (jCol = 0; jCol < endingWidth; jCol = jCol + windowWidth) {
        getROI.x = jCol;
        getROI.y = iRow;
        getROI.width = windowWidth;
        getROI.height = (gray_image.rows - iRow);
        Mat croppedImg = gray_image(getROI);

        prepareBlocks.startX = getROI.x;
        prepareBlocks.startY = getROI.y;
        prepareBlocks.endX = getROI.x + getROI.width;
        prepareBlocks.endY = getROI.y + getROI.height;
        prepareBlocks.croppedImg = croppedImg;

/*        cv::Rect rect(getROI.x, getROI.y, getROI.width, getROI.height);
        cv::rectangle(gray_image, rect, cv::Scalar(0, 255, 0));*/

        Scalar meanWindowVal;
        Scalar stdWindowVal;
        meanStdDev(croppedImg, meanWindowVal, stdWindowVal);
        double pickMeanVal = meanWindowVal.val[0];
        double pickStdVal = stdWindowVal.val[0];

        prepareBlocks.stdWindowVal = pickStdVal;
        prepareBlocks.meanWindowVal = pickMeanVal;

        if (pickStdVal < minStandardDev)  // getting the minimum standard deviation
            minStandardDev = pickStdVal;

        if (pickStdVal > maxStandardDev)  // getting the maximum standard Deviation
            maxStandardDev = pickStdVal;
        keepAllBlockImag.push_back(prepareBlocks);
    }
  //  BasicAlgo::getInstance()->showImage(gray_image);

    // For the end row
    for (iRow = 0; iRow < endingHeight; iRow = iRow + windowHeight) {

        // get the last block image
        getROI.x = jCol;
        getROI.y = iRow;
        getROI.width = (gray_image.cols - jCol);;
        getROI.height = windowHeight;
        Mat croppedImg = gray_image(getROI);

        prepareBlocks.startX = getROI.x;
        prepareBlocks.startY = getROI.y;
        prepareBlocks.endX = getROI.x + getROI.width;
        prepareBlocks.endY = getROI.y + getROI.height;
        prepareBlocks.croppedImg = croppedImg;

/*        cv::Rect rect(getROI.x, getROI.y, getROI.width, getROI.height);
        cv::rectangle(gray_image, rect, cv::Scalar(0, 255, 0));*/

        Scalar meanWindowVal;
        Scalar stdWindowVal;
        meanStdDev(croppedImg, meanWindowVal, stdWindowVal);
        double pickMeanVal = meanWindowVal.val[0];
        double pickStdVal = stdWindowVal.val[0];

        prepareBlocks.stdWindowVal = pickStdVal;
        prepareBlocks.meanWindowVal = pickMeanVal;

        if (pickStdVal < minStandardDev)  // getting the minimum standard deviation
            minStandardDev = pickStdVal;

        if (pickStdVal > maxStandardDev)  // getting the maximum standard Deviation
            maxStandardDev = pickStdVal;
        keepAllBlockImag.push_back(prepareBlocks);
    }
   // BasicAlgo::getInstance()->showImage(gray_image);

    // for the bottom right block
    getROI.x = jCol;
    getROI.y = iRow;
    getROI.width = (gray_image.cols - jCol);;
    getROI.height = (gray_image.rows - iRow);
    Mat croppedImg = gray_image(getROI);

    prepareBlocks.startX = getROI.x;
    prepareBlocks.startY = getROI.y;
    prepareBlocks.endX = getROI.x + getROI.width;
    prepareBlocks.endY = getROI.y + getROI.height;
    prepareBlocks.croppedImg = croppedImg;

/*    cv::Rect rect(getROI.x, getROI.y, getROI.width, getROI.height);
    cv::rectangle(gray_image, rect, cv::Scalar(0, 255, 0));*/

    Scalar meanWindowVal;
    Scalar stdWindowVal;
    meanStdDev(croppedImg, meanWindowVal, stdWindowVal);
    double pickMeanVal = meanWindowVal.val[0];
    double pickStdVal = stdWindowVal.val[0];

    prepareBlocks.stdWindowVal = pickStdVal;
    prepareBlocks.meanWindowVal = pickMeanVal;

    if (pickStdVal < minStandardDev)  // getting the minimum standard deviation
        minStandardDev = pickStdVal;

    if (pickStdVal > maxStandardDev)  // getting the maximum standard Deviation
        maxStandardDev = pickStdVal;
    keepAllBlockImag.push_back(prepareBlocks);
   // BasicAlgo::getInstance()->showImage(gray_image);

}


/** @brief ﻿An adaptive local binarization method for document images based on a novel thresholding method and dynamic windows :
 * ﻿Bilal Bataineh, Siti Norul Huda Sheikh Abdullah, Khairuddin Omar Center
 *
 *  Doing the binarization of the grey scale image
 *
 *  @param Grey scale image.
 *  @param Blank image of the same size as grey scale source image.
 *  @return Void.
 *  @author Spandan
 *  @bug No known bugs.
 *  @date 12/12/2018
 */

void PopularDocBinarization::binarizeDynamicWindows(Mat &gray_image, Mat &binarImag) {
    vector<blockImagProperties> keepAllBlockImag;

    double minImage, maxImage;
    cv::minMaxLoc(gray_image, &minImage, &maxImage);


    // Global mean of the image
    cv::Scalar tempMeanVal = mean(gray_image);
    float globalMean = tempMeanVal.val[0];

    // Global std of the image
    Scalar justDumpMean, getGlobalStdDev;
    meanStdDev(gray_image, justDumpMean, getGlobalStdDev);

    float globalSTD = getGlobalStdDev.val[0];
    float Tcon = globalMean -
                 ((pow(globalMean, 2) * globalSTD) / ((globalMean + globalSTD) * ((0.5 * maxImage) + globalSTD)));

    float leftLimit = Tcon - (globalSTD / 2);
    float rightLimit = Tcon + (globalSTD / 2);

    Mat newConfusionImg = Mat::zeros(gray_image.rows, gray_image.cols, gray_image.type());

    int numRedPix = 0;
    int numBlackPixel = 0;
    int numWhitePixel = 0;
    for (int ii = 0; ii < gray_image.rows; ii++) {
        for (int jj = 0; jj < gray_image.cols; jj++) {
            if ((gray_image.at<uchar>(ii, jj) < leftLimit) || (gray_image.at<uchar>(ii, jj) == leftLimit)) {
                newConfusionImg.at<uchar>(ii, jj) = 0;  // for black
                numBlackPixel++;
            } else if ((gray_image.at<uchar>(ii, jj) > leftLimit) && (gray_image.at<uchar>(ii, jj) < rightLimit)) {
                newConfusionImg.at<uchar>(ii, jj) = 25; // for red
                numRedPix++;
            } else if ((gray_image.at<uchar>(ii, jj) > rightLimit) || (gray_image.at<uchar>(ii, jj) == rightLimit)) {
                newConfusionImg.at<uchar>(ii, jj) = 255; // for white
                numWhitePixel++;
            }
        }
    }

    // the probability of red and black pixels
    float getProb = (float) numBlackPixel / (float) numRedPix;
    int winHeight = 0;
    int winWidth = 0;

    if ((getProb > 2.5) || (getProb == 2.5) || globalSTD < (0.1 * maxImage)) {
        winHeight = (gray_image.rows / 4);
        winWidth = (gray_image.cols / 6);
    } else if (((getProb > 1) && (getProb < 2.5)) || ((gray_image.rows + gray_image.cols) < 400)) {
        winHeight = (gray_image.rows / 20);
        winWidth = (gray_image.cols / 30);
    } else if ((getProb < 1) || (getProb == 1)) {
        winHeight = (gray_image.rows / 30);
        winWidth = (gray_image.cols / 40);
    }


    // get all the block image
    int windowHeight = winHeight;
    int windowWidth = winWidth;

    double minStandardDev = 1000000000.0;
    double maxStandardDev = 0.0;

    divideImageIntoSubImages(gray_image, windowHeight, windowWidth, keepAllBlockImag, minStandardDev, maxStandardDev);
    binarImag = Mat::zeros(gray_image.rows, gray_image.cols, gray_image.type());


    for (int ii = 0; ii < keepAllBlockImag.size(); ii++) {

       // cout << "I am at block : " << ii << endl;
        int startX = keepAllBlockImag.at(ii).startX;
        int endX = keepAllBlockImag.at(ii).endX;

        int startY = keepAllBlockImag.at(ii).startY;
        int endY = keepAllBlockImag.at(ii).endY;


        int tempNumRed = 0;
        int tempNumBlack = 0;
        for (int subWinRw = startY; subWinRw < endY; subWinRw++) {
            for (int subWinCol = startX; subWinCol < endX; subWinCol++) {
                if (newConfusionImg.at<uchar>(subWinRw, subWinCol) == 25) // it is red
                    tempNumRed++;
                else if (newConfusionImg.at<uchar>(subWinRw, subWinCol) == 0) // it is black
                    tempNumBlack++;
            }
        }

        if (tempNumRed > tempNumBlack) {
            int getSubTempWinHeight = endY - startY;
            int getSubTempWinWidth = endX - startX;


            int topLeft_X_Start = startX;
            int topLeft_Width = (getSubTempWinWidth / 2);

            int topLeft_Y_Start = startY;
            int topLeft_Height = (getSubTempWinHeight / 2);
            getBinarizedImageSubSubParts(gray_image, binarImag, topLeft_X_Start, topLeft_Width, topLeft_Y_Start,
                                         topLeft_Height,
                                         minStandardDev, maxStandardDev, maxImage, globalMean);


            int topRight_X_Start = startX + (getSubTempWinWidth / 2);
            int topRight_Width = endX - topRight_X_Start;

            int topRight_Y_Start = startY;
            int topRight_Height = (getSubTempWinHeight / 2);
            getBinarizedImageSubSubParts(gray_image, binarImag, topRight_X_Start, topRight_Width, topRight_Y_Start,
                                         topRight_Height, minStandardDev, maxStandardDev, maxImage, globalMean);


            int bottomLeft_X_Start = startX;
            int bottomLeft_Width = (getSubTempWinWidth / 2);

            int bottomLeft_Y_Start = startY + (getSubTempWinHeight / 2);
            int bottomLeft_Height = endY - bottomLeft_Y_Start;
            getBinarizedImageSubSubParts(gray_image, binarImag, bottomLeft_X_Start, bottomLeft_Width,
                                         bottomLeft_Y_Start,
                                         bottomLeft_Height, minStandardDev, maxStandardDev, maxImage, globalMean);


            int bottomRight_X_Start = startX + (getSubTempWinWidth / 2);
            int bottomRight_Width = endX - topRight_X_Start;

            int bottomRight_Y_Start = startY + (getSubTempWinHeight / 2);
            int bottomRight_Height = endY - bottomLeft_Y_Start;
            getBinarizedImageSubSubParts(gray_image, binarImag, bottomRight_X_Start, bottomRight_Width,
                                         bottomRight_Y_Start,
                                         bottomRight_Height, minStandardDev, maxStandardDev, maxImage, globalMean);


        } else {
            Mat getSubSubImag = keepAllBlockImag.at(ii).croppedImg;
            Scalar meanWindowVal;
            Scalar stdWindowVal;
            meanStdDev(getSubSubImag, meanWindowVal, stdWindowVal);
            double pickMeanVal = meanWindowVal.val[0];
            double sigmaW = stdWindowVal.val[0];

            double sigmaAdaptive = ((sigmaW - minStandardDev) / (maxStandardDev - minStandardDev)) * maxImage;
            double Twindow = pickMeanVal - (((pickMeanVal * pickMeanVal) * sigmaW) /
                                            ((globalMean + sigmaW) * (sigmaAdaptive + sigmaW)));

            // for the inside image loop
            for (int iR = startY; iR < endY; iR++) {
                for (int jC = startX; jC < endX; jC++) {
                    if (gray_image.at<uchar>(iR, jC) < Twindow)
                        binarImag.at<uchar>(iR, jC) = 0;
                    else if (gray_image.at<uchar>(iR, jC) >= Twindow)
                        binarImag.at<uchar>(iR, jC) = 255;
                }
            }
        }


    }
   // BasicAlgo::getInstance()->showImage(binarImag);
}



void PopularDocBinarization::GetSomeStatistics_From_FullImage(Mat& completeGreyImage,
                                                              float& globalMean, float& globalSTD, double& minImage, double& maxImage ){

    // YOU HAVE TO WORK AGIAN HERE

    int startTextCol = 0, endTextCol = completeGreyImage.cols, startTextRow = 0, endTextRow = completeGreyImage.rows;
    cv::Rect getROI;

    getROI.x = startTextCol;
    getROI.width = (endTextCol - startTextCol);
    getROI.y = startTextRow;
    getROI.height = (endTextRow - startTextRow);

    Mat validGrayImageArea = completeGreyImage(getROI);
    // BasicAlgo::getInstance()->showImage(validGrayImageArea);


    // Get the min and max of the image
    cv::minMaxLoc(validGrayImageArea, &minImage, &maxImage);

    // Global mean of the image
    cv::Scalar tempMeanVal = mean(validGrayImageArea);
    globalMean = tempMeanVal.val[0];

    // Global std of the image
    Scalar justDumpMean, getGlobalStdDev;
    meanStdDev(validGrayImageArea, justDumpMean, getGlobalStdDev);
    globalSTD = getGlobalStdDev.val[0];

}


void PopularDocBinarization::SimpleDynamicWindowBased_Binarization(Mat& subGreyImage, Mat& subBinaryImage, Mat& ResultBinImag){

    float globalMean = 0.0, globalSTD = 0.0;
    double minImage = 0.0, maxImage = 0.0;
    GetSomeStatistics_From_FullImage(subGreyImage, globalMean, globalSTD, minImage, maxImage);

    int numOfPartition = 20;
    int partitionSize = (int)(round(subGreyImage.cols / 20));
    blockImagProperties prepareBlocks;
    vector<blockImagProperties> keepAllBlockImag;

    cv::Rect getROI;
    int startBlock_x = 0;
    double minStandardDev = 1000000000.0;
    double maxStandardDev = 0.0;

    for (int partMe = 0; partMe < (numOfPartition-1); partMe++){

        getROI.x = startBlock_x;
        getROI.width = partitionSize;
        getROI.y = 0;
        getROI.height = subGreyImage.rows;

        Mat croppedImg = subGreyImage(getROI);
        prepareBlocks.startX = getROI.x;
        prepareBlocks.startY = getROI.y;
        prepareBlocks.endX = getROI.x + getROI.width;
        prepareBlocks.endY = getROI.y + getROI.height;
        prepareBlocks.croppedImg = croppedImg;

        Scalar meanWindowVal;
        Scalar stdWindowVal;
        meanStdDev(croppedImg, meanWindowVal, stdWindowVal);
        double pickMeanVal = meanWindowVal.val[0];
        double pickStdVal = stdWindowVal.val[0];

        prepareBlocks.stdWindowVal = pickStdVal;
        prepareBlocks.meanWindowVal = pickMeanVal;

        keepAllBlockImag.push_back(prepareBlocks);

        if (pickStdVal < minStandardDev)  // getting the minimum standard deviation
            minStandardDev = pickStdVal;

        if (pickStdVal > maxStandardDev)  // getting the maximum standard Deviation
            maxStandardDev = pickStdVal;

        startBlock_x = startBlock_x + partitionSize;
    }

    // For the very last part of image
    getROI.x = startBlock_x;
    getROI.width = partitionSize;
    getROI.y = 0;
    getROI.height = subGreyImage.rows;

    Mat croppedImg = subGreyImage(getROI);
    prepareBlocks.startX = getROI.x;
    prepareBlocks.startY = getROI.y;
    prepareBlocks.endX = getROI.x + getROI.width;
    prepareBlocks.endY = getROI.y + getROI.height;
    prepareBlocks.croppedImg = croppedImg;

    Scalar meanWindowVal;
    Scalar stdWindowVal;
    meanStdDev(croppedImg, meanWindowVal, stdWindowVal);
    double pickMeanVal = meanWindowVal.val[0];
    double pickStdVal = stdWindowVal.val[0];

    prepareBlocks.stdWindowVal = pickStdVal;
    prepareBlocks.meanWindowVal = pickMeanVal;

    keepAllBlockImag.push_back(prepareBlocks);

    if (pickStdVal < minStandardDev)  // getting the minimum standard deviation
        minStandardDev = pickStdVal;

    if (pickStdVal > maxStandardDev)  // getting the maximum standard Deviation
        maxStandardDev = pickStdVal;


    ResultBinImag = Mat(subGreyImage.size(), subGreyImage.type(), Scalar(255));
    for (int iPartImg = 0; iPartImg < keepAllBlockImag.size(); iPartImg++){
        Mat getSubSubImag = keepAllBlockImag.at(iPartImg).croppedImg;
        double pickMeanVal = keepAllBlockImag.at(iPartImg).meanWindowVal;
        double sigmaW = keepAllBlockImag.at(iPartImg).stdWindowVal;
        double sigmaAdaptive = ((sigmaW - minStandardDev) / (maxStandardDev - minStandardDev)) * maxImage;
        double Twindow = pickMeanVal - (((pickMeanVal * pickMeanVal) * sigmaW) /
                                        ((globalMean + sigmaW) * (sigmaAdaptive + sigmaW)));

        int startY = keepAllBlockImag.at(iPartImg).startY;
        int endY = keepAllBlockImag.at(iPartImg).endY;
        int startX = keepAllBlockImag.at(iPartImg).startX;
        int endX = keepAllBlockImag.at(iPartImg).endX;

        for (int iR = startY; iR < endY; iR++) {
            for (int jC = startX; jC < endX; jC++) {
                if (subGreyImage.at<uchar>(iR, jC) < Twindow)
                    ResultBinImag.at<uchar>(iR, jC) = 0;
                else if (subGreyImage.at<uchar>(iR, jC) >= Twindow)
                    ResultBinImag.at<uchar>(iR, jC) = 255;
            }
        }


    }

}
void PopularDocBinarization::getBinarizedImageSubSubParts(Mat& gray_image, Mat& binImage, int xStart, int width,
                                                          int yStart, int height, double minStandardDev, double maxStandardDev,
                                                           double maxImage, float globalMean ){
        cv::Rect getROI;

        getROI.x = xStart;
        getROI.y = yStart;
        getROI.width = width;
        getROI.height = height;

        Mat getSubSubImag = gray_image(getROI);

        Scalar meanWindowVal;
        Scalar stdWindowVal;
        meanStdDev(getSubSubImag, meanWindowVal, stdWindowVal);
        double pickMeanVal = meanWindowVal.val[0];
        double sigmaW = stdWindowVal.val[0];

        double sigmaAdaptive = ((sigmaW - minStandardDev) / (maxStandardDev - minStandardDev)) * maxImage;
        double Twindow = pickMeanVal - (  ((pickMeanVal* pickMeanVal) * sigmaW) /
                                          (   (globalMean + sigmaW) * (sigmaAdaptive + sigmaW)  )   )   ;


        // for the inside image loop
        for (int iR = yStart; iR < (yStart + height); iR++) {
            for (int jC = xStart; jC < (xStart + width); jC++) {
                if (gray_image.at<uchar>(iR, jC) < Twindow)
                    binImage.at<uchar>(iR, jC) = 0;
                else if ((gray_image.at<uchar>(iR, jC) > Twindow) || (gray_image.at<uchar>(iR, jC) == Twindow) )
                    binImage.at<uchar>(iR, jC) = 255;
            }
        }
}

/** @brief Article : Efficient Implementation of Local Adaptive Thresholding Techniques Using Integral Images : Faisal Shafait, Daniel Keysers, Thomas M. Breuel
 *
 *  Doing the binarization of the grey scale image
 *
 *  @param Grey scale image.
 *  @param Blank image of the same size as grey scale source image.
 *  @return Void.
 *  @author Spandan
 *  @bug No known bugs.
 *  @date 13/4/2016
 */
 
void PopularDocBinarization::binarizeSafait(Mat &gray_image,Mat &bin_image){
    whalf = w>>1;
    int x = 0,y = 0;
    if(k < 0.05 && k > 0.95)
        fprintf(stderr,"[sauvola %g %d]\n",k,w);
    if(w < 0 && k > 1000)
        fprintf(stderr,"[sauvola %g %d]\n",k,w);
    
    //            if(bin_image.length1d()!=gray_image.length1d())
    //                makelike(bin_image,gray_image);
    
    // if the grey scale image is only containing 0 and 255 then copy the grey image in binary image and return it
    //            if(contains_only(gray_image,byte(0),byte(255))){
    //                copy(bin_image,gray_image);
    //                return ;
    //            }
    
    int image_width  = gray_image.cols;  // X coordinate
    int image_height = gray_image.rows;  // Y coordinate
    whalf = w>>1;
    
    // Calculate the integral image, and integral of the squared image
    long long int **integral_image,**rowsum_image,**integral_sqimg,**rowsum_sqimg;
    
    // dynaically allocating the memory to the array
    
    integral_image = new long long int*[image_height];
    rowsum_image = new long long int*[image_height];
    integral_sqimg = new long long int*[image_height];
    rowsum_sqimg = new long long int*[image_height];
    
    for(y = 0;y < gray_image.rows; y++)
    {
        integral_image[y] = new long long int[image_width];
        rowsum_image[y] = new long long int[image_width];
        integral_sqimg[y] = new long long int[image_width];
        rowsum_sqimg[y] = new long long int[image_width];
    }
    for (y = 0; y < image_height; y++){  // representing x coordinate
        for (x = 0; x < image_width; x++){  // representing y coordinate
            integral_image[y][x] = 0;
            rowsum_image[y][x] = 0;
            integral_sqimg[y][x] = 0;
            rowsum_sqimg[y][x] = 0;
            
        }
    }
    
    // for initializing the array with zeros
//    for (int i = 0; i < image_height*image_width; i++){
//        *((int*)integral_image + i) = 0;
//        *((int*)rowsum_image + i) = 0;
//        *((int*)integral_sqimg + i) = 0;
//        *((int*)rowsum_sqimg + i) = 0;
//    }
    
    int xmin,ymin,xmax,ymax;
    double diagsum,idiagsum,diff,sqdiagsum,sqidiagsum,sqdiff,area;
    double mean,std,threshold;
    
    // performing square and original for the first row of the image
    for(y=0; y<image_height; y++){
        rowsum_image[y][0] = gray_image.at<uchar>(y,0);
        rowsum_sqimg[y][0] = gray_image.at<uchar>(y,0) * gray_image.at<uchar>(y,0); // squaring operation
    }
    
    //doing integral and squared summations for the full image top to bottom
    for(y = 1; y < image_height; y++){  // for each row i.e. Y directoin
        for(x = 0; x < image_width; x++){ // for each column i.e. X direction
            rowsum_image[y][x] = rowsum_image[y-1][x] + gray_image.at<uchar>(y,x); // doing the simple intregation i.e. summing up the top to bottom of the image
            rowsum_sqimg[y][x] = rowsum_sqimg[y-1][x] + gray_image.at<uchar>(y,x) * gray_image.at<uchar>(y,x); // making the squared image
        }
    }
    
    for(x = 0; x < image_width; x++){
        integral_image[0][x] = rowsum_image[0][x];  // coping the previously attained rowsum_sqimg in the variable integral_image
        integral_sqimg[0][x] = rowsum_sqimg[0][x];  // coping the previously attained rowsum_sqimg in the variable integral_sqimg
    }
    
    // generatingv the integral image
    for(y = 0; y < image_height; y++){  // for each row i.e. Y directoin
        for(x = 1; x < image_width; x++){ // for each column i.e. X direction
            integral_image[y][x] = integral_image[y][x-1] + rowsum_image[y][x];
            integral_sqimg[y][x] = integral_sqimg[y][x-1] + rowsum_sqimg[y][x];
        }
    }
    releasingDynamicArray(gray_image.rows, rowsum_image);
    releasingDynamicArray(gray_image.rows, rowsum_sqimg);
    
    //Calculate the mean and standard deviation using the integral image
    
    for(y = 0; y < image_height; y++){  // for each row i.e. Y directoin
        for(x = 1; x < image_width; x++){ // for each column i.e. X direction
            
            // *******  This portion of code is to calculate the threshold   *********************
            xmin = max(0,x-whalf);
            ymin = max(0,y-whalf);
            xmax = min(image_width-1,x+whalf);
            ymax = min(image_height-1,y+whalf);
            area = (xmax-xmin+1)*(ymax-ymin+1);
            
            
            assert(area);  //  If this area evaluates to 0, this causes an assertion failure that terminates the program.
            if(!xmin && !ymin){ // Point at origin  xmin = ymin = 0; (not 0)  = 1
                diff   = integral_image[ymax][xmax];
                sqdiff = integral_sqimg[ymax][xmax];
            }
            else if(!xmin && ymin){ // first row xmin = 0
                diff   = integral_image[ymax][xmax] - integral_image[ymax-1][xmin];
                sqdiff = integral_sqimg[ymax][xmax] - integral_sqimg[ymax-1][xmin];
            }
            else if(xmin && !ymin){ // first col  ymin = 0
                diff   = integral_image[ymax][xmax] - integral_image[ymin][xmax-1];
                sqdiff = integral_sqimg[ymax][xmax] - integral_sqimg[ymin][xmax-1];
            }
            else{ // rest of the image
                diagsum    = integral_image[ymax][xmax] + integral_image[ymin-1][xmin-1];
                idiagsum   = integral_image[ymax][xmin-1] + integral_image[ymin-1][xmax];
                diff       = diagsum - idiagsum;
                sqdiagsum  = integral_sqimg[ymax][xmax] + integral_sqimg[ymin-1][xmin-1];
                sqidiagsum = integral_sqimg[ymax][xmin-1] + integral_sqimg[ymin-1][xmax];
                sqdiff     = sqdiagsum - sqidiagsum;
            }
            
            mean = diff/area;
            std  = sqrt((sqdiff - diff*diff/area)/(area-1));
            threshold = mean*(1+k*((std/128)-1));
            //  **************                ****************************************************
            
            
            if(gray_image.at<uchar>(y,x) < threshold)
                bin_image.at<uchar>(y,x) = 0;
            else
                bin_image.at<uchar>(y,x) = MAXVAL-1;
            
        }
    }
    
    releasingDynamicArray(gray_image.rows, integral_image);
    releasingDynamicArray(gray_image.rows, integral_sqimg);
}
// pass the "key" and "value"
void PopularDocBinarization::set(const char *key,double value) {
    if(!strcmp(key,"k")) this->k = value;
    else if(!strcmp(key,"w")) this->w = int(value);
    else throw "unknown parameter";
}

/*******
 *  The normalization technique is taken from this paper
 *
﻿Jia, F., Shi, C., He, K., Wang, C., & Xiao, B. (2018). Degraded document image binarization using structural symmetry of strokes.
Pattern Recognition, 74, 225–240.

 *******/
void PopularDocBinarization::Do_Another_Image_Normalization(Mat& origImg, Mat& backGdImag, Mat& I_norm) {
    Mat F_xy = 255 * (origImg / backGdImag);
    I_norm = Mat::zeros(origImg.rows, origImg.cols, origImg.type());

    for (int ii = 0; ii < origImg.rows; ii++)
        for (int jj = 0; jj < origImg.cols; jj++)
            if ( (origImg.at<uchar>(ii,jj) < backGdImag.at<uchar>(ii,jj)) && (backGdImag.at<uchar>(ii,jj) > 0) )
                I_norm.at<uchar>(ii,jj) =  F_xy.at<uchar>(ii,jj);
            else
                I_norm.at<uchar>(ii,jj) = 255;


    BasicAlgo::getInstance()->showImage(I_norm);
}


/*******
 *  The normalization technique is taken from this paper
 *
﻿Jia, F., Shi, C., He, K., Wang, C., & Xiao, B. (2018). Degraded document image binarization using structural symmetry of strokes.
Pattern Recognition, 74, 225–240.

 *******/
void PopularDocBinarization::Gradient_Image_Calculation(Mat& I_norm, Mat& gradImag) {

    int kernel_size = 3;
    float dataHori[9] = { -3, 0, 3, -10, 0, 10, -3, 0, 3 };
    float dataVerti[9] = { -3, 10, -3, 0, 0, 0, 3, 10, 3 };

    cv::Mat kernelHori = cv::Mat(3, 3, CV_32F, dataHori);
    cv::Mat kernelVerti = cv::Mat(3, 3, CV_32F, dataVerti);

    /// Initialize arguments for the filter
    cv::Point anchor = Point( -1, -1 );
    double delta = 0;
    int ddepth = -1;

    /// Apply filter
    Mat I_norm_hori, I_norm_verti;
    filter2D(I_norm, I_norm_hori, ddepth , kernelHori, anchor, delta, BORDER_DEFAULT );
    filter2D(I_norm, I_norm_verti, ddepth , kernelVerti, anchor, delta, BORDER_DEFAULT );

    gradImag = Mat::zeros(I_norm.rows, I_norm.cols, CV_16U);
    gradImag = I_norm_hori + I_norm_verti;

    // BasicAlgo::getInstance()->showImage(gradImag);
    normalize( gradImag, gradImag, 255, 0);
    // BasicAlgo::getInstance()->showImage(gradImag);
}

/*******
 *  The normalization technique is taken from this paper
 *
﻿Jia, F., Shi, C., He, K., Wang, C., & Xiao, B. (2018). Degraded document image binarization using structural symmetry of strokes.
Pattern Recognition, 74, 225–240.

 *******/
void PopularDocBinarization::SSP_Extraction( Mat& gradImag, Mat& G_t_xy) {

    vector< pair <int,double> > threshPairs;
    G_t_xy = Mat::zeros(gradImag.rows, gradImag.cols, gradImag.type());
    Mat F_Edge_G_t_xy = Mat::zeros(gradImag.rows, gradImag.cols, gradImag.type());

    float x1, y1;

    for (int th = 0; th < 256; th++){


        for (int ii = 0; ii < gradImag.rows; ii++){
            for (int jj = 0; jj < gradImag.cols; jj++){
                if(gradImag.at<uchar>(ii,jj) > th) {
                    G_t_xy.at<uchar>(ii, jj) = 1;
                    bool enterFlagNeigh = false;

                    for(int k = -1; (k <= 1) && !enterFlagNeigh; k++) {
                        for (int j = -1; (j <= 1) && !enterFlagNeigh; j++) {
                            x1 = reflect(gradImag.cols, jj - j);
                            y1 = reflect(gradImag.rows, ii - k);

                            if (gradImag.at<uchar>(y1, x1) == 0)
                                enterFlagNeigh = true;
                        }
                    }
                    if(enterFlagNeigh)
                        F_Edge_G_t_xy.at<uchar>(ii, jj) = 1;
                }
                else {
                    G_t_xy.at<uchar>(ii, jj) = 0;
                }
            }
        }


        // calculate number of connected component
        int numComp;
        BasicProcessingTechniques::getInstance()->connectedComponentLabeling(G_t_xy,numComp);
        double sum = cv::sum(G_t_xy)[0];
        double thresh = sum / numComp;

        threshPairs.emplace_back( make_pair(th, thresh) );

    }
}



void PopularDocBinarization::OrigImageNormalization_by_BackGdImage(Mat& origImg, Mat& backGdImag, Mat& N_xy) {
    Mat F_xy = (origImg + 1) / (backGdImag + 1);
    double min_F_xy;
    double max_F_xy;
    minMaxIdx(F_xy, &min_F_xy, &max_F_xy);

    double min_origImg;
    double max_origImg;
    minMaxIdx(origImg, &min_origImg, &max_origImg);

    N_xy = (max_origImg - min_origImg) * ((F_xy - min_F_xy) / (max_F_xy + min_F_xy)) + (min_origImg);
   // BasicAlgo::getInstance()->showImage(N_xy);
}


void PopularDocBinarization::CreateBackGroundImage(Mat& im, Mat& backGdImag) {
    Mat copyOrigImg = im;

    int winx = 60, winy = 60;
    Scalar value;
    RNG rng(12345);
    Mat imPadded;
    value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    copyMakeBorder(im, imPadded, (winy / 2), (winy / 2), (winx / 2), (winx / 2), BORDER_REPLICATE, value);

    Mat backGdMasks = Mat::zeros(imPadded.rows, imPadded.cols, imPadded.type());
    NiblackSauvolaWolfJolion (imPadded, backGdMasks, NIBLACK, 60, 60, -0.2, 128);
    Mat backGdMasksCropped;
    imPadded.release();
    backGdMasksCropped = backGdMasks(Rect((winx/2), (winy/2), im.cols, im.rows));
    backGdMasks.release();

    Mat dialationSE = getStructuringElement(MORPH_RECT, Size(3, 3));   // Define the structuring elements
    Mat dialatedBackGdMasks;

    Mat backGdMasksComplement = BasicProcessingTechniques::getInstance()->ComplementImage(backGdMasksCropped.clone());

    morphologyEx(backGdMasksComplement, dialatedBackGdMasks, MORPH_DILATE, dialationSE);
    backGdMasksCropped.release();
    backGdMasksComplement.release();
    Mat dialatedBackGdComplement = BasicProcessingTechniques::getInstance()->ComplementImage(dialatedBackGdMasks.clone());
    dialatedBackGdMasks = dialatedBackGdComplement;
    dialatedBackGdComplement.release();
    dialatedBackGdMasks = dialatedBackGdMasks/255;  // making the image on 0 and 1


    int X_Start[] = {0, 0, copyOrigImg.cols, copyOrigImg.cols};
    int X_End[] = {copyOrigImg.cols, copyOrigImg.cols, 0, 0};
    int Y_Start[] = {0, copyOrigImg.rows, 0, copyOrigImg.rows};
    int Y_End[] = {copyOrigImg.rows, 0, copyOrigImg.rows, 0};

    backGdImag = Mat::zeros(copyOrigImg.rows, copyOrigImg.cols, copyOrigImg.type());
    vector<Mat>Pimages;

    /****   First Image     ***/
    Mat copyDialatedBackGdMasks = dialatedBackGdMasks.clone();
    Mat temp_P_Image = Mat::zeros(copyOrigImg.rows, copyOrigImg.cols, CV_8U);
    for (int yy = (Y_Start[0]+1); yy < (Y_End[0]-1); yy++)
        for (int xx = (X_Start[0]+1); xx < (X_End[0]-1); xx++)
            DoBackGroundImageOperation(copyOrigImg,copyDialatedBackGdMasks, temp_P_Image, yy, xx);

    Pimages.push_back(temp_P_Image);
    temp_P_Image.release();


    /****   Second Image     ***/
    copyDialatedBackGdMasks = dialatedBackGdMasks.clone();
    temp_P_Image = Mat::zeros(copyOrigImg.rows, copyOrigImg.cols, CV_8U);
    for (int yy = (Y_Start[1]-2); yy > (Y_End[1]); yy--) // as the last index is 498 if for example image has 500 rows
        for (int xx = (X_Start[1]+1); xx < (X_End[1]-1); xx++) // here also it goes until 498 if it has 500 cols  (< (500-1) = 499)
            DoBackGroundImageOperation(copyOrigImg,copyDialatedBackGdMasks, temp_P_Image, yy, xx);

    Pimages.push_back(temp_P_Image);
    temp_P_Image.release();


    /****   Third Image     ***/
    copyDialatedBackGdMasks = dialatedBackGdMasks.clone();
    temp_P_Image = Mat::zeros(copyOrigImg.rows, copyOrigImg.cols, CV_8U);
    for (int yy = (Y_Start[2]+1); yy < (Y_End[2]-1); yy++) {
        for (int xx = (X_Start[2] - 2); xx > (X_End[2]); xx--) {
            DoBackGroundImageOperation(copyOrigImg, copyDialatedBackGdMasks, temp_P_Image, yy, xx);
        }
    }
    Pimages.push_back(temp_P_Image);
    temp_P_Image.release();


    /****   Fourth Image     ***/
    copyDialatedBackGdMasks = dialatedBackGdMasks.clone();
    temp_P_Image = Mat::zeros(copyOrigImg.rows, copyOrigImg.cols, CV_8U);
    for (int yy = (Y_Start[3]-2); yy > (Y_End[3]); yy--) {
        cout << yy << endl;
        for (int xx = (X_Start[3] - 2); xx > (X_End[3]); xx--) {
            DoBackGroundImageOperation(copyOrigImg, copyDialatedBackGdMasks, temp_P_Image, yy, xx);
        }
    }
    Pimages.push_back(temp_P_Image);
    temp_P_Image.release();
    copyDialatedBackGdMasks.release();
    dialatedBackGdMasks.release();
    copyOrigImg.release();


    for (int yy = Y_Start[0]; yy < Y_End[0]; yy++) {
        for (int xx = X_Start[0]; xx < X_End[0]; xx++) {
            int minValue = (int)std::min(std::min (Pimages.at(0).at<uchar>(yy,xx),Pimages.at(1).at<uchar>(yy,xx)) ,
                                    std::min (Pimages.at(2).at<uchar>(yy,xx),Pimages.at(3).at<uchar>(yy,xx)) );
            //cout << minValue << endl;
            backGdImag.at<uchar>(yy, xx) = (uchar)minValue;
        }
    }
    BasicAlgo::getInstance()->showImage(backGdImag);
}

inline void PopularDocBinarization::DoBackGroundImageOperation(Mat& copyOrigImg, Mat& dialatedBackGdMasks, Mat& temp_P_Image, int yy, int xx){
    if(dialatedBackGdMasks.at<uchar>(yy,xx) == 0){
        int avgVal = (int)(  (copyOrigImg.at<uchar>(yy,xx-1) *  dialatedBackGdMasks.at<uchar>(yy,xx-1)) +
                             (copyOrigImg.at<uchar>(yy-1,xx) * dialatedBackGdMasks.at<uchar>(yy-1,xx)) +
                             (copyOrigImg.at<uchar>(yy,xx+1) * dialatedBackGdMasks.at<uchar>(yy,xx+1)) +
                             (copyOrigImg.at<uchar>(yy+1,xx) * dialatedBackGdMasks.at<uchar>(yy+1,xx))  ) / 4 ;

        temp_P_Image.at<uchar>(yy,xx) = (uchar)avgVal;
        copyOrigImg.at<uchar>(yy,xx) = (uchar)avgVal;
        dialatedBackGdMasks.at<uchar>(yy,xx) = 1;
    }
}







void PopularDocBinarization::Calculate_Mean_StandardDeviation_Features(Mat& im, Mat& map_m, Mat& map_s, int winx, int winy) {

    Scalar value;
    RNG rng(12345);
    Mat imPadded;
    value = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    copyMakeBorder(im, imPadded, (winy / 2), (winy / 2), (winx / 2), (winx / 2), BORDER_REPLICATE, value);

    // Create local statistics and store them in a double matrices
    Mat map_m_padded = Mat::zeros(imPadded.rows, imPadded.cols, CV_32F);
    Mat map_s_padded = Mat::zeros(imPadded.rows, imPadded.cols, CV_32F);
    double max_s = calcLocalStats(imPadded, map_m_padded, map_s_padded, winx, winy);
    map_m = map_m_padded(Rect((winx/2), (winy/2), im.cols, im.rows));
    map_s = map_s_padded(Rect((winx/2), (winy/2), im.cols, im.rows));

    Mat map_m_norm;
    Mat map_s_norm;
    normalize( map_m, map_m_norm, 1, 0, NORM_MINMAX, CV_32F, Mat() );
    normalize( map_s, map_s_norm, 1, 0, NORM_MINMAX, CV_32F, Mat() );

    // BasicAlgo::getInstance()->showImage(map_m_norm);
    // BasicAlgo::getInstance()->showImage(map_s_norm);
}



void PopularDocBinarization::CalculateNiblackFeatures (Mat& im, Mat& featureMat, int winx, int winy) {

    double min_I, max_I;

    Scalar value;
    RNG rng(12345);
    Mat imPadded;
    value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
    copyMakeBorder( im, imPadded, (winy/2), (winy/2), (winx/2), (winx/2), BORDER_REPLICATE, value );

    // BasicAlgo::getInstance()->showImage(imPadded);

    // Create local statistics and store them in a double matrices
    Mat map_m_padded = Mat::zeros (imPadded.rows, imPadded.cols, CV_32F);
    Mat map_s_padded = Mat::zeros (imPadded.rows, imPadded.cols, CV_32F);
    double max_s = calcLocalStats (imPadded, map_m_padded, map_s_padded, winx, winy);
    Mat map_m = map_m_padded(Rect((winx/2), (winy/2), im.cols, im.rows));
    Mat map_s = map_s_padded(Rect((winx/2), (winy/2), im.cols, im.rows));




    minMaxLoc(im, &min_I, &max_I);
    Mat thsurf (im.rows, im.cols, CV_32F);

    featureMat = Mat::zeros (im.rows, im.cols, CV_32F);

    for (int ii  = 0; ii < im.rows; ii++){
        for (int jj = 0; jj < im.cols; jj++){
            float tempVal = ((im.at<uchar>(ii,jj) - map_m.at<float>(ii,jj)) / map_s.at<float>(ii,jj));
            float exponentialNiblack = exp(tempVal);
            if (im.at<uchar>(ii,jj) <= map_m.at<float>(ii,jj)){
                featureMat.at<float>(ii,jj) = exponentialNiblack ;
            }
            else {
                featureMat.at<float>(ii,jj) = 1.0;
            }
        }
    }
//    BasicAlgo::getInstance()->showImage(featureMat);
}


void PopularDocBinarization::CalculateSavoulaFeatures (Mat& im, Mat& featureMat, int winx, int winy, float S_savoula) {

    double min_I, max_I;

    Scalar value;
    RNG rng(12345);
    Mat imPadded;
    value = Scalar( rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255) );
    copyMakeBorder( im, imPadded, (winy/2), (winy/2), (winx/2), (winx/2), BORDER_REPLICATE, value );

    // BasicAlgo::getInstance()->showImage(imPadded);

    // Create local statistics and store them in a double matrices
    Mat map_m_padded = Mat::zeros (imPadded.rows, imPadded.cols, CV_32F);
    Mat map_s_padded = Mat::zeros (imPadded.rows, imPadded.cols, CV_32F);
    double max_s = calcLocalStats (imPadded, map_m_padded, map_s_padded, winx, winy);
    Mat map_m = map_m_padded(Rect((winx/2), (winy/2), im.cols, im.rows));
    Mat map_s = map_s_padded(Rect((winx/2), (winy/2), im.cols, im.rows));


    minMaxLoc(im, &min_I, &max_I);
    Mat thsurf (im.rows, im.cols, CV_32F);

    featureMat = Mat::zeros (im.rows, im.cols, CV_32F);
    // double K_savoula = 0.5;
    for (int ii  = 0; ii < im.rows; ii++){
        for (int jj = 0; jj < im.cols; jj++){
            float tempVal =  ((im.at<uchar>(ii,jj) / map_m.at<float>(ii,jj)) -1 )  / ((map_s.at<float>(ii,jj) / S_savoula) - 1) ;

            float exponentialNiblack = exp(-tempVal);
            if (map_s.at<float>(ii,jj) > S_savoula){
                featureMat.at<float>(ii,jj) = 0 ;
            } else {
                auto denoVal = (1.0 + exponentialNiblack);
                auto tempValGoFor = pow(denoVal,(-1));
                featureMat.at<float>(ii,jj) = (float)tempValGoFor;
            }
        }
    }
    BasicAlgo::getInstance()->showImage(featureMat);
}

int PopularDocBinarization::reflect(int M, int x)
{
    if(x < 0)
    {
        return -x - 1;
    }
    if(x >= M)
    {
        return 2*M - x - 1;
    }
    return x;
}

void PopularDocBinarization::LogIntensityPercentileFeatures (Mat& im, Mat& featureMat, int winx, int winy) {
    float sum, x1, y1;
    float Th_perc = 0.01;

    featureMat = Mat::zeros (im.rows, im.cols, CV_32F);

    for(int y = 0; y < im.rows; y++){
        for(int x = 0; x < im.cols; x++){
            sum = 0.0;
            int s_cnt = 0;
            for(int k = -(winy/2); k <= (winy/2); k++){
                for(int j = -(winx/2); j <= (winx/2); j++ ){
                    x1 = reflect(im.cols, x - j);
                    y1 = reflect(im.rows, y - k);
                    int tempVal = (im.at<uchar>(y,x) - im.at<uchar>(y1,x1));
                    int binVal;
                    if(tempVal >= 0)
                        binVal = 1;
                    else
                        binVal = 0;
                    sum = sum + binVal;
                    s_cnt++;
                }
            }
            sum = sum / s_cnt;
            if (sum <= Th_perc)
                featureMat.at<float>(y,x) = 1;
            else
                featureMat.at<float>(y,x) = log(sum) / log(Th_perc);
        }
    }
//    BasicAlgo::getInstance()->showImage(featureMat);
}




void PopularDocBinarization::RelativeDarknessIndexFeatures (Mat& im, Mat& featureMat, int relaxation) {

    float x1, y1;

    featureMat = Mat::zeros (im.rows, im.cols, CV_32F);

    Mat featureMat_RDI_1 = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat featureMat_RDI_0 = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat featureMat_RDI_minus_1 = Mat::zeros (im.rows, im.cols, CV_32F);

    Mat featureMat_RDI_1_div_0_1 = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat featureMat_RDI_0_div_minus_1_0 = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat featureMat_RDI_minus_1_div_minus_1_1 = Mat::zeros (im.rows, im.cols, CV_32F);

    for(int y = 0; y < im.rows; y++){
        for(int x = 0; x < im.cols; x++){

            int codeCnt_1 = 0;
            int codeCnt_0 = 0;
            int codeCnt_minus_1 = 0;

            for(int k = -1; k <= 1; k++){
                for(int j = -1; j <= 1; j++ ){
                    if ((k!= 0) || (j != 0) ) {  // just to avoid the central pixel

                        int codeVal;
                        x1 = reflect(im.cols, x - j);
                        y1 = reflect(im.rows, y - k);
                        if ( ((im.at<uchar>(y1, x1)) > (im.at<uchar>(y, x) + relaxation)) ||
                                ((im.at<uchar>(y1, x1)) > (im.at<uchar>(y, x) + relaxation)) ) {
                            codeVal = 1;
                        } else if ( ((im.at<uchar>(y1, x1)) < (im.at<uchar>(y, x) - relaxation)) ||
                                    ((im.at<uchar>(y1, x1)) == (im.at<uchar>(y, x) - relaxation)) ){
                            codeVal = -1;
                        } else if ((abs(im.at<uchar>(y1, x1) - im.at<uchar>(y, x))) < relaxation) {
                            codeVal = 0;
                        }
                        // for code :  1
                        if ((codeVal - 1) == 0) // that means the codeVal = 1
                            codeCnt_1 = codeCnt_1 + 1;
                            // for code : 0
                        else if ((codeVal - 0) == 0)    // that means the codeVal = 1
                            codeCnt_0 = codeCnt_0 + 1;
                            // for code : -1
                        else if ((codeVal - (-1)) == 0)    // that means the codeVal = 1
                            codeCnt_minus_1 = codeCnt_minus_1 + 1;
                    }
                }
            }
            featureMat_RDI_1.at<float>(y,x) = ((float)codeCnt_1 / 8);
            featureMat_RDI_0.at<float>(y,x) = ((float)codeCnt_0 / 8);
            featureMat_RDI_minus_1.at<float>(y,x) = ((float)codeCnt_minus_1 / 8);

            featureMat_RDI_1_div_0_1.at<float>(y,x) = featureMat_RDI_1.at<float>(y,x) /
                                                (featureMat_RDI_0.at<float>(y,x) + featureMat_RDI_1.at<float>(y,x));

            featureMat_RDI_0_div_minus_1_0.at<float>(y,x) = featureMat_RDI_0.at<float>(y,x) /
                                                (featureMat_RDI_minus_1.at<float>(y,x) + featureMat_RDI_0.at<float>(y,x));

            featureMat_RDI_minus_1_div_minus_1_1.at<float>(y,x) = featureMat_RDI_minus_1.at<float>(y,x) /
                                                (featureMat_RDI_minus_1.at<float>(y,x) + featureMat_RDI_1.at<float>(y,x)) ;

        }
    }
    BasicAlgo::getInstance()->showImage(featureMat_RDI_1);
    BasicAlgo::getInstance()->showImage(featureMat_RDI_0);
    BasicAlgo::getInstance()->showImage(featureMat_RDI_minus_1);

    BasicAlgo::getInstance()->showImage(featureMat_RDI_1_div_0_1);
    BasicAlgo::getInstance()->showImage(featureMat_RDI_0_div_minus_1_0);
    BasicAlgo::getInstance()->showImage(featureMat_RDI_minus_1_div_minus_1_1);
}


void PopularDocBinarization::BolanSUFeatures (Mat& im, Mat& featureMat, int windowXFull, int windowYFull) {
    float sum, x1, y1;
    float Th_perc = 0.01;

    float epsilonSU = 0.001;
    int windowX = windowXFull/2;
    int windowY = windowYFull/2;

    Mat contrastImg  = Mat::zeros(im.rows, im.cols, CV_32F);

    for(int y = 0; y < im.rows; y++){
        for(int x = 0; x < im.cols; x++){
            vector<int> keepAllEle;
            for(int k = -windowY;k <= windowY; k++){
                for(int j = -windowX;j <= windowX; j++ ){
                    x1 = reflect(im.cols, x - j);
                    y1 = reflect(im.rows, y - k);
                    int tempVal = (im.at<uchar>(y1,x1));
                    keepAllEle.push_back(tempVal);
                }
            }
            auto result = std::minmax_element(keepAllEle.begin(), keepAllEle.end());

            auto minVal = keepAllEle.at(result.first - keepAllEle.begin());
            auto maxVal = keepAllEle.at(result.second - keepAllEle.begin());

            auto C_ij = (maxVal - minVal) / ((maxVal + minVal) + epsilonSU);
            contrastImg.at<float>(y,x) = C_ij;
        }
    }
    BasicAlgo::getInstance()->showImage(contrastImg);
}

void PopularDocBinarization::printFloatMatrix(cv::Mat imgMat) {
    for (int i = 0; i < imgMat.rows; i++) {
        for (int j = 0; j < imgMat.cols; j++) {

            cout << (float) imgMat.at<float>(i, j) << " ";
        }
        cout << endl;
    }
}

void PopularDocBinarization::HoweFeatures (Mat& im, Mat& howeFeature, int windowXFull, int windowYFull) {

    Mat abs_dst, dst;
    int kernel_size = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    Laplacian( im, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    convertScaleAbs( dst, abs_dst );

    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros (im.rows, im.cols, CV_32F);
    Mat map_s = Mat::zeros (im.rows, im.cols, CV_32F);
    double max_s = calcLocalStats (im, map_m, map_s, windowXFull, windowYFull);
    howeFeature  = Mat::zeros(im.rows, im.cols, CV_32F);

    for(int ii = 0; ii < im.rows; ii++) {
        for (int jj = 0; jj < im.cols; jj++) {
            //       if ((abs_dst.at<uchar>(ii,jj) > 0) && (map_m.at<float>(ii,jj) > 0) ) {
                float getVal = (abs_dst.at<uchar>(ii, jj) * map_m.at<float>(ii, jj));
                howeFeature.at<float>(ii,jj) = getVal;
     //           cout << getVal << endl;
    //        }
        }
    }
}

int PopularDocBinarization::CalculateStrokeWidth(Mat& gray_image,  Mat& strokeImage){

    vector<int>keepAllPairDist;


    for(int y = 0; y < strokeImage.rows; y++) {
        vector<cv::Point> storIntensityCoord;
        bool myEnterFlag = false;
        for (int x = 0; x < (strokeImage.cols - 1); x++) {
            if( (strokeImage.at<uchar>(y,x) == 0) && (strokeImage.at<uchar>(y,x+1) == 255) ) {
              int pixIntensity = gray_image.at<uchar>(y,x);
              int pixIntensityNext = gray_image.at<uchar>(y,x+1);
              if(pixIntensity > pixIntensityNext){ // taking only those pixels whose intensity is more than the next pixel intensity
                  cv::Point putPoints;
                  putPoints.x = x;
                  putPoints.y = y;
                  storIntensityCoord.push_back(putPoints);
                  myEnterFlag = true;
              }
            }
        }
        if(myEnterFlag) {
            for (int kVec = 0; kVec < (storIntensityCoord.size() - 1); kVec++) {
                float dist = euclideanDist(storIntensityCoord.at(kVec), storIntensityCoord.at(kVec + 1));
                keepAllPairDist.push_back(dist);
            }
        }
        storIntensityCoord.clear();
    }



    vector<int> assembleGoodEleRefined;
    int avgBlackRunLength = BasicProcessingTechniques::getInstance()->callGetMyMeanValUpdated(keepAllPairDist,
            keepAllPairDist.size(), assembleGoodEleRefined);
    return avgBlackRunLength;
}

void PopularDocBinarization::CalculateFeaturesForBinarization(Mat &gray_image, Mat &bin_image) {


/*    Mat backGroundImag;
    CreateBackGroundImage(gray_image, backGroundImag);
    Mat normalizedOrigImag;
    OrigImageNormalization_by_BackGdImage(gray_image, backGroundImag, normalizedOrigImag);*/


/*    Mat meanFeature, stdFeature;
    Calculate_Mean_StandardDeviation_Features(gray_image, meanFeature, stdFeature, 8, 8);*/


/*    Mat bulanSuFeatures;
    BolanSUFeatures(gray_image, bulanSuFeatures, 8, 8);*/


/*    Mat howeFeatures;
    HoweFeatures(gray_image, howeFeatures, 3, 3);
    normalize(howeFeatures, howeFeatures, 255, 0);
    BasicAlgo::getInstance()->showImage(howeFeatures);*/
    //normalize( howeFeatures, howeFeatures, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );


/*    Mat niBlackFeatures;
    CalculateNiblackFeatures(gray_image, niBlackFeatures, 64, 64);*/


/*    Mat savoulaFeatures;
    float S_savoula = 128.0;
    CalculateSavoulaFeatures(gray_image, savoulaFeatures, 8, 8, S_savoula);*/


    Mat logPercentFeatures;
    LogIntensityPercentileFeatures(gray_image, logPercentFeatures, 64, 64);
    BasicAlgo::getInstance()->showImage(logPercentFeatures);

    Mat RDI_FeatureMat;
    RelativeDarknessIndexFeatures(gray_image, RDI_FeatureMat, 5);



    Scalar meanWindowVal;
    Scalar stdWindowVal;
    meanStdDev(gray_image, meanWindowVal, stdWindowVal);
    double sigmaW = stdWindowVal.val[0];
    float gamma = 0.85;
    float alpha = pow(((float) sigmaW / 128), gamma);

    float x1, y1;

    float epsilonSU = 0.001;
    int windowX = 1;
    int windowY = 1;

    Mat contrastImg = Mat::zeros(gray_image.rows, gray_image.cols, CV_32F);
    Mat modContrastImg = Mat::zeros(gray_image.rows, gray_image.cols, CV_32F);
    Mat modContrastImgChange = Mat::zeros(gray_image.rows, gray_image.cols, CV_8U);

    for (int y = 0; y < gray_image.rows; y++) {
        for (int x = 0; x < gray_image.cols; x++) {
            vector<int> keepAllEle;
            for (int k = -windowY; k <= windowY; k++) {
                for (int j = -windowX; j <= windowX; j++) {
                    x1 = reflect(gray_image.cols, x - j);
                    y1 = reflect(gray_image.rows, y - k);
                    int tempVal = (gray_image.at<uchar>(y1, x1));
                    keepAllEle.push_back(tempVal);
                }
            }
            auto result = std::minmax_element(keepAllEle.begin(), keepAllEle.end());

            auto minVal = keepAllEle.at((result.first - keepAllEle.begin()));
            auto maxVal = keepAllEle.at((result.second - keepAllEle.begin()));

            float C_ij = (maxVal - minVal) / ((maxVal + minVal) + epsilonSU);

            float C_a_ij = (alpha * C_ij) + ((1 - alpha) * ((maxVal - minVal) / 255));

            contrastImg.at<float>(y, x) = C_ij;
            modContrastImg.at<float>(y, x) = C_a_ij;
            modContrastImgChange.at<uchar>(y, x) = (uchar) (255 * C_a_ij);
        }
    }
    // BasicAlgo::getInstance()->showImage(modContrastImgChange) ;

/*    normalize(modContrastImg, modContrastImgChange, 255,0);
    BasicAlgo::getInstance()->showImage(modContrastImg);
    BasicAlgo::getInstance()->printMatrix(modContrastImgChange);
    printFloatMatrix(modContrastImgChange);*/

    Mat modContrastImgBin;
    double otsuThreshVal = cv::threshold(modContrastImgChange, modContrastImgBin, 0, 255,
                                         CV_THRESH_BINARY | CV_THRESH_OTSU);
    // BasicAlgo::getInstance()->showImage(modContrastImgBin);

    Mat cannyImage;
    double highThreshCanny = otsuThreshVal;
    double lowThreshCanny = otsuThreshVal * 0.5;
    Canny(gray_image, cannyImage, lowThreshCanny, highThreshCanny, 3);
    // BasicAlgo::getInstance()->showImage(cannyImage);

    Mat bitwiseAndImg;
    bitwise_and(modContrastImgBin, cannyImage, bitwiseAndImg);
    // BasicAlgo::getInstance()->showImage(bitwiseAndImg);

    CalculateStrokeWidth(gray_image, bitwiseAndImg);
}
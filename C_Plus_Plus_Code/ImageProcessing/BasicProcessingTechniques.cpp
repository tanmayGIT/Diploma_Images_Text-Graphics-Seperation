/*
 * BasicProcessingTechniques.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: tanmoymondal
 */

#include "./hdr/BasicProcessingTechniques.h"

namespace std {
    using namespace cv;
    using namespace std;
    BasicProcessingTechniques *BasicProcessingTechniques::instance = 0;

    BasicProcessingTechniques *BasicProcessingTechniques::getInstance() {
        if (!instance)
            instance = new BasicProcessingTechniques();

        return instance;
    }

    BasicProcessingTechniques::BasicProcessingTechniques() {
    }

    BasicProcessingTechniques::~BasicProcessingTechniques() {
    }

    struct sortOpenCVPoints {
        bool operator()(cv::Point pt1, cv::Point pt2) { return (pt1.x < pt2.x); }
    } sortingObject;

    bool compareLess(cv::Point a, cv::Point b, cv::Point center)
    {
        if (a.x - center.x >= 0 && b.x - center.x < 0)
            return true;
        if (a.x - center.x < 0 && b.x - center.x >= 0)
            return false;
        if (a.x - center.x == 0 && b.x - center.x == 0) {
            if (a.y - center.y >= 0 || b.y - center.y >= 0)
                return a.y > b.y;
            return b.y > a.y;
        }

        // compute the cross product of vectors (center -> a) x (center -> b)
        int det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
        if (det < 0)
            return true;
        if (det > 0)
            return false;

        // points a and b are on the same line from the center
        // check which point is closer to the center
        int d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
        int d2 = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (b.y - center.y);
        return d1 > d2;
    }


    int BasicProcessingTechniques::callGetMyMeanValUpdated(vector<int> &keepWhiteRunLengthsVec, int whiteRunCnt,
                                                           std::vector<int> &assembleGoodEleRefined) {
        int avgWhiteRun;
        // putting the vector into array form
        int *keepWhiteRunLengths = NULL;
        keepWhiteRunLengths = new int[whiteRunCnt];

        if (keepWhiteRunLengthsVec.size() == whiteRunCnt) {
            for (int i = 0; i < whiteRunCnt; i++) {
                keepWhiteRunLengths[i] = keepWhiteRunLengthsVec.at(i);
            }
        } else {
            assert("There is some issue becuase the lengh of the vector and white run should be same");
        }
        try {
            if (keepWhiteRunLengthsVec.size() > 5)
                //                avgWhiteRun = getAdvancedMeanVal(keepWhiteRunLengths,(int) keepWhiteRunLengthsVec.size()); // was giving some bugs
                avgWhiteRun = getMyMeanValUpdated(keepWhiteRunLengths, (int) keepWhiteRunLengthsVec.size(),
                                                  assembleGoodEleRefined);

        } catch (exception &e) {
            cout << e.what() << '\n';
        }
        return avgWhiteRun;
    }



    int BasicProcessingTechniques::getMyMeanValUpdated(int *arrOfEle, int arr_sz,
                                                       std::vector<int> &assembleGoodEleRefined) {

        int *sorted_data_height = quicksort(arrOfEle, 0, (arr_sz));
        int maxCompHeight = sorted_data_height[arr_sz -
                                               1]; // as the array is sorted, so the maximum value will be in the last cell
        int minCompHeight = sorted_data_height[0]; // the minimum value will at the initial index

        int noOfBins = 10;
        vector<BinBucketDivision> keepAllBucketInfo;
        if ((maxCompHeight - minCompHeight) > 50) // then only enter in the loop for doing some binning operation
        {
            int binWidth = (maxCompHeight - minCompHeight) / noOfBins; // you can change this value 10 later
            BinBucketDivision binBuckDiv;
            // Defining the bin's limits and values
            for (int i = 0; i < noOfBins; i++) {
                binBuckDiv.binLowerBound = minCompHeight + (binWidth * i);
                binBuckDiv.binUpperBound = minCompHeight + (binWidth * (i + 1));
                keepAllBucketInfo.push_back(binBuckDiv);
            }
            // Making the following vectors of the same size as the one of "keepAllBucketInfo"
            vector<vector<int>> keepBucketVals(keepAllBucketInfo.size());
            vector<vector<int>> keepBucketIndexes(keepAllBucketInfo.size());

            //calculate histogram array
            for (int i = 0; i < arr_sz; i++) {
                int getMyVal = sorted_data_height[i];

                int binApprox;
                if (getMyVal > binWidth) {
                    int getDivRemainder = (getMyVal % binWidth);
                    if (getDivRemainder > 0) {
                        // the bin estimation would be one more
                        binApprox = (getMyVal / binWidth) + 1;
                    } else {
                        binApprox = (getMyVal / binWidth);
                    }
                    if ((((binWidth * binApprox) > getMyVal) || ((binWidth * binApprox) == getMyVal)) &&
                        (((binWidth * (binApprox - 1)) < getMyVal) || ((binWidth * (binApprox - 1)) == getMyVal))) {

                        if (binApprox > noOfBins)
                            binApprox = noOfBins;
                        if (binApprox < 1)
                            binApprox = 1;

                        keepBucketIndexes.at(binApprox - 1).push_back(i);
                        keepBucketVals.at(binApprox - 1).push_back(getMyVal);
                    } else {
                        assert("There is a problem");
                    }
                } else {
                    binApprox = 0;
                    keepBucketIndexes.at(binApprox).push_back(i);
                    keepBucketVals.at(binApprox).push_back(getMyVal);
                }
            }
            std::vector<std::pair<int, int> > binSz;
            int cntEleBinVect = 0;
            for (auto &getEleBinVect : keepBucketIndexes) {
                binSz.push_back(std::make_pair((int) getEleBinVect.size(), cntEleBinVect));
                cntEleBinVect++;
            }
            std::sort(binSz.begin(),
                      binSz.end()); // we have sorted in ascending order so the last 2 entries here would be largest

            int consideredVal = 2; // how many bins you are considering
            std::vector<int> assembleGoodEle;
            for (int pp = 0; pp < consideredVal; pp++) {
                // pick from the end of the vector
                int numOfEle = binSz.at(noOfBins - (pp + 1)).first; // no. of elements
                int indexToLookFor = binSz.at(noOfBins - (pp + 1)).second; // no. of elements
                for (int pickElements = 0; pickElements < numOfEle; pickElements++) {
                    assembleGoodEle.push_back(keepBucketVals.at(indexToLookFor).at(pickElements));
                }
            }

            int meanValofGoodEle = std::accumulate(std::begin(assembleGoodEle), std::end(assembleGoodEle), 0.0);
            meanValofGoodEle = meanValofGoodEle / assembleGoodEle.size();

            double accum = 0.0;
            std::for_each(std::begin(assembleGoodEle), std::end(assembleGoodEle), [&](const double d) {
                accum += (d - meanValofGoodEle) * (d - meanValofGoodEle);
            });

            double getStDev = sqrt(accum / (assembleGoodEle.size() - 1));
            for (int i = 0; i < arr_sz; i++) {
                int getMyVal = sorted_data_height[i];

                if (((meanValofGoodEle - getStDev) <= getMyVal) && (getMyVal <= (meanValofGoodEle + getStDev))) {
                    if (getMyVal > 8) // this is a threshold, we put by heuristic and this value should be changed
                        assembleGoodEleRefined.push_back(getMyVal);
                }
            }
            // see whether by putting the criteria of 8, we could have some elements in the vector or not. If not then remove this criteria and again obtain the array.
            if (assembleGoodEleRefined.size() < 3) {
                for (int i = 0; i < arr_sz; i++) {
                    int getMyVal = sorted_data_height[i];

                    if (((meanValofGoodEle - getStDev) <= getMyVal) && (getMyVal <= (meanValofGoodEle + getStDev))) {
                        assembleGoodEleRefined.push_back(getMyVal);
                    }
                }
            }
        } else {
            for (int i = 0; i < arr_sz; i++) {
                assembleGoodEleRefined.push_back(sorted_data_height[i]);
            }
        }
        int meanValofGoodEleRefined = std::accumulate(std::begin(assembleGoodEleRefined),
                                                      std::end(assembleGoodEleRefined), 0.0);
        meanValofGoodEleRefined = meanValofGoodEleRefined / assembleGoodEleRefined.size();
        return meanValofGoodEleRefined;
    }




    inline Mat BasicProcessingTechniques::GetAverageRowValue(Mat &binImage) {
        for (int i = 0; i < binImage.rows; i++) {
            int sumVals = 0;
            for (int j = 0; j < binImage.cols; j++) {
                sumVals = sumVals + (int) binImage.at<uchar>(i, j);
            }
            binImage.row(i).setTo(Scalar(round(sumVals / binImage.cols)));
        }
        return binImage;
    }


    /**
     *   @brief  Perform horizontal RLSA of the component image
     *
     *   @param  The binary image
     *   @param  The threshold for horizontal RLSA
     */
    Mat BasicProcessingTechniques::Horizontal_RLSATechnique(Mat &bin_image, int hor_thres) {
        Mat bin_img_RLSA = bin_image;
        int zero_count = 0;
        int one_flag = 0;
        for (int i = 0; i < ((bin_image.rows)); i++) {
            for (int j = hor_thres; j < (bin_image.cols) - hor_thres; j++) {
                if (bin_image.at<uchar>(i, j) == 255) {
                    if (one_flag == 255) {
                        if (zero_count <= hor_thres) {
                            bin_img_RLSA(cv::Range(i, i + 1), cv::Range(j - zero_count, j)).setTo(cv::Scalar::all(255));
                        } else {
                            one_flag = 0;
                        }
                        zero_count = 0;
                    }
                    one_flag = 255;
                } else {
                    if (one_flag == 255) {
                        zero_count = zero_count + 1;
                    }
                }
            }
        }
        return bin_img_RLSA;
    }



    /**
     *   @brief  Perform the connected component labeling
     *
     *   @param  The binary image
     *   @param  The number of components
     */
    ComponentInfo *BasicProcessingTechniques::connectedComponentLabeling(Mat bin_image, int &numComponent) {
        // connected component labeling
        // for this funciton you need black background and white foreground
        Rect bounding_rect;
        vector<vector<Point> > contours; // Vector for storing contour
        vector<Vec4i> hierarchy;
        Mat dst(bin_image.rows, bin_image.cols, CV_8UC1, Scalar::all(0));
        findContours(bin_image, contours, hierarchy, CV_RETR_CCOMP,
                     CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
        ComponentInfo *cominfo = new ComponentInfo[contours.size()];
        for (unsigned int i = 0; i < contours.size(); i++) // iterate through each contour.
        {
            cominfo[i].xstart = boundingRect(contours[i]).x;
            cominfo[i].ystart = boundingRect(contours[i]).y;
            cominfo[i].height = boundingRect(contours[i]).height;
            cominfo[i].width = boundingRect(contours[i]).width;
            cominfo[i].xcenter = cominfo[i].xstart + (cominfo[i].width / 2);
            cominfo[i].ycenter = cominfo[i].ystart + (cominfo[i].height / 2);
            cominfo[i].area = contourArea(contours[i], false);  //  Find the area of contour
        }
        numComponent = static_cast<int>(contours.size());
        return cominfo;
    }

    /**
 *   @brief   To get the connected component's statistics

 *   @param  The PPP binary image
 *   @return All the blobs from the image and corresponding statistics
 */

    void BasicProcessingTechniques::getConnectedCompOperationPPPImage(Mat &binPPPimage,
                                                                      vector<ComponentInfo> &wholeComInfo,
                                                                      vector<bool> &labelCompHeirarchy,
                                                                      int &avgHeight, int &avgWidth) {
        //BasicAlgo::getInstance()->showImage(binPPPimage);
        vector<vector<Point>> contourPPPImage;
        vector<Vec4i> hierarchy;
        Mat labels(binPPPimage.size(), CV_32S);

        findContours(binPPPimage.clone(), contourPPPImage, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image

        // get the moments
        vector<Moments> mu(contourPPPImage.size());
        // get the centroid of components.
        vector<Point2f> mc(contourPPPImage.size());

        vector<int> keepAllHeight;
        vector<int> keepAllWidth;
        int count = 1; //0 is background
        for (unsigned int i = 0; i < contourPPPImage.size(); i++) // iterate through each contour.
        {
            ComponentInfo tempComp;
            tempComp.xstart = boundingRect(contourPPPImage[i]).x;
            tempComp.ystart = boundingRect(contourPPPImage[i]).y;
            tempComp.height = boundingRect(contourPPPImage[i]).height;
            tempComp.width = boundingRect(contourPPPImage[i]).width;

            mu[i] = moments(contourPPPImage[i], false);
            mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);

            tempComp.xcenter = std::round(mc[i].x);
            tempComp.ycenter = std::round(mc[i].y);
            tempComp.area = contourArea(contourPPPImage[i], false);
            if (hierarchy[i][3] == -1) // Getting only the outer contour of each blobs
            {
                keepAllHeight.push_back(tempComp.height);
                keepAllWidth.push_back(tempComp.width);
                // Thickness of lines the contours are drawn with. If it is negative (for example, thickness=CV_FILLED ), the contour interiors are drawn.
                drawContours(labels, contourPPPImage, i, Scalar(count), CV_FILLED, 8, hierarchy, 0, Point()); // 5th param : thickness –

                vector<cv::Point> singleBlob;
                singleBlob.reserve(tempComp.width * tempComp.height);
                for (size_t xCol = tempComp.xstart; xCol < (tempComp.xstart + tempComp.width); xCol++) {
                    for (size_t yRow = tempComp.ystart; yRow < (tempComp.ystart + tempComp.height); yRow++) {
                        Point generatePoint((int) xCol, (int) yRow);
                        if (count == labels.at<int>(generatePoint)) // picking only the on pixels
                        {
                            singleBlob.push_back(generatePoint);
                        }
                    }
                }
                tempComp.outerBlobPts = singleBlob;
                labelCompHeirarchy.push_back(true);
                count++;
            } else {
                labelCompHeirarchy.push_back(false);
            }
            wholeComInfo.push_back(tempComp);

        }
        std::vector<int> goodAllHeightRefined;
        avgHeight = callGetMyMeanValUpdated(keepAllHeight, keepAllHeight.size(), goodAllHeightRefined);
        std::vector<int> goodAllWidthRefined;
        avgWidth = callGetMyMeanValUpdated(keepAllWidth, keepAllWidth.size(), goodAllWidthRefined);
    }

    /**
     *   @brief  Doing the complement of the image
     *
     *   @param  The structure, containing the information of the components
     */
    Mat BasicProcessingTechniques::ComplementImage(Mat binImageCopy) {
        // so we need to complement the image
        Mat new_image = Mat::zeros(binImageCopy.size(), binImageCopy.type());
        Mat complement_bin_mat = Mat::ones(binImageCopy.size(), binImageCopy.type()) * 255;
        subtract(complement_bin_mat, binImageCopy, new_image);
        return new_image;
    }



    /**
     *   @brief  Cropping the image by the given boundary
     *
     *   @param  The IplImage pointer
     *   @param  The ROI for cropping
     */
    IplImage *crop(IplImage *src, CvRect roi) {

        // Must have dimensions of output image
        IplImage *cropped = cvCreateImage(cvSize(roi.width, roi.height), src->depth, src->nChannels);

        // Say what the source region is
        cvSetImageROI(src, roi);

        // Do the copy
        cvCopy(src, cropped);
        cvResetImageROI(src);

        //	cvNamedWindow( "check", 1 );
        //	cvShowImage( "check", cropped );
        //	cvSaveImage ("style.jpg" , cropped);

        return cropped;
    }





} /* namespace std */

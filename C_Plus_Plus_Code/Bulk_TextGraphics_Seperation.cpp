//
//  TextGraphicsSeperation.cpp
//  DocScanImageProcessing
//
//  Created by tmondal on 16/07/2018.
//  Copyright © 2018 Tanmoy. All rights reserved.
//


/*
#include <stdio.h>
#include "hdr/LogoDetectionTechnique.hpp"
#include "ImageFiltering/hdr/AnisotropicGaussianFilter.h"
#include "ImageFiltering/hdr/AdaptiveNonlocalFiltering.h"

#include "ImageProcessing/SLIC_SuperPIxels/SLIC_Original.h"
#include "ImageProcessing/hdr/NoisyBorderRemoval.hpp"

#include <opencv2/ximgproc.hpp>
#include <ImageProcessing/hdr/SlicBy_OpenCV.h>
#include "opencv2/core/utility.hpp"
#include "tesseract/baseapi.h"
#include <highgui.h>
#include <cv.h>
#include <omp.h>

using namespace cv::ximgproc;

Size kernalSize (5,5);
RNG rng(12345);
int main(int argc, char** argv) {
    DirectoryHandler* instance = DirectoryHandler::getInstance();
    cv::Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection("/Users/tmondal/Documents/Workspace_C++/Text_Graphics_Seperation/DocScanImageProcessing/model.yml");
    string allImgDir = "/Users/tmondal/Documents/Datasets/Univ_Diploma/BDD3_JPEG/scan fujitsu/";

    string resultSavingDir = "/Users/tmondal/Documents/Datasets/Univ_Diploma/All_Result_Dir/";
    // Get the image files
    static vector<string> validImgFileNames;
    static vector<string> validImgExtensions;
    validImgExtensions.push_back("jpg");
    if(boost::filesystem::exists( allImgDir ))
        instance->getFilesInDirectory(allImgDir, validImgFileNames, validImgExtensions);



// #pragma omp parallel
    for (auto const& imgFilePath: validImgFileNames) {
        boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
        string onlyNm = getImgPath.filename().string(); //  getting only the file name
        onlyNm.erase(onlyNm.find_last_of("."), string::npos);

        string imgFullPath = allImgDir + getImgPath.stem().string() + ".jpg";

        Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);
        if (imgOrig.empty()) {
            std::cerr << "File not available for reading" << std::endl;
           // return -1;
        }
        Mat imgGrey = imgOrig.clone(); // keeping seperately the original image


        //   **********************************   Doing Adaptive Offset Filtering **************************************

        // AdaptiveNonlocalFiltering doAdaptiveFiltering;
        // doAdaptiveFiltering.offsetFilter(imgGrey,"Gaussian");
        //   **********************************   End of Adaptive Offset Filtering **************************************


        string newDirPath = resultSavingDir + onlyNm;
        if (! (boost::filesystem::exists(newDirPath)))
            boost::filesystem::create_directory(newDirPath);
        string customSavingPath = newDirPath +  "/" + "1_original_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(imgGrey, customSavingPath);
        auto start = chrono::steady_clock::now();
        if (imgGrey.channels() >= 3)
            cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);


        //   **********************************   Doing Random Forest Based Edge Detection ********************************
        Mat colorImg;
        cv::cvtColor(imgGrey, colorImg, cv::COLOR_GRAY2BGR);
        Mat3b imgGreyCopy = (Mat3b) colorImg.clone();
        colorImg.release();

        Mat3f f_imgGrey;
        imgGreyCopy.convertTo(f_imgGrey, CV_32F, 1.0 / 255.0); // Convert source image to [0;1] range
        imgGreyCopy.release();

        Mat1f edgeImage;
        pDollar->detectEdges(f_imgGrey, edgeImage);
        f_imgGrey.release();

        // computes orientation from edge map
        Mat orientation_map;
        pDollar->computeOrientation(edgeImage, orientation_map);
        // suppress edges
        Mat edge_nms;
        pDollar->edgesNms(edgeImage, orientation_map, edge_nms, 2, 0, 1, true);
        edgeImage = 255 * edgeImage;
        BasicAlgo::getInstance()->writeImage(edgeImage);
        customSavingPath = newDirPath +  "/" + "2_gradient_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(edgeImage, customSavingPath);
        orientation_map.release();
        edgeImage.release();
        //   **********************************   End of Random Forest Based Edge Detection ********************************


        Mat gray_image = imread("/Users/tmondal/Documents/Save_Image.jpg", CV_LOAD_IMAGE_GRAYSCALE);
        Mat myOutputBin = Mat::zeros(gray_image.rows, gray_image.cols, CV_8U);
        PopularDocBinarization::getInstance()->binarizeSafait(imgGrey, myOutputBin);
        BasicAlgo::getInstance()->writeImageGivenPath(myOutputBin,"/Users/tmondal/Documents/2_bin_Image.jpg");
        myOutputBin.release();



        // *****               Anisotropic diffusion filter                    *****  //
        Mat filteredImage = gray_image.clone();
        filteredImage.convertTo(filteredImage, CV_32FC1);

        AnisotropicGaussianFilter filtImag = *new AnisotropicGaussianFilter();
        filtImag.doAnisotropicDiffusionFiltering(filteredImage, filteredImage.cols, filteredImage.rows);
        double min;
        double max;
        minMaxIdx(filteredImage, &min, &max);
        filteredImage.convertTo(filteredImage, CV_8UC1, 255 / (max - min), -min);
        customSavingPath = newDirPath +  "/" + "3_filtered_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, customSavingPath);



        // *****               Apply Canny Edge Detection                   *****  //
        Mat dummyMat;
        double otsuThreshVal = cv::threshold(filteredImage, dummyMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        dummyMat.release(); // delete the dummy mat
        double highThreshCanny = otsuThreshVal;
        double lowThreshCanny = otsuThreshVal * 0.5;
        Canny(filteredImage, filteredImage, lowThreshCanny, highThreshCanny, 3);
        customSavingPath = newDirPath +  "/" + "4_canny_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, customSavingPath);

        // filteredImage = ImageFiltering::getInstance()->removeSmallNoisesDots(filteredImage);
        // BasicAlgo::getInstance()->writeImageGivenPath(filteredImage,"/Users/tmondal/Documents/5_small_noiseRemoved_Image.jpg");

        // Perform dilation operation
        Mat dialationSE = getStructuringElement(MORPH_RECT, Size(7, 7));   // Define the structuring elements
        Mat dialatedImg;
        morphologyEx(filteredImage, dialatedImg, MORPH_DILATE, dialationSE); // Perform dilation operation
        filteredImage.release();
        customSavingPath = newDirPath +  "/" + "5_canny_dialted_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(dialatedImg, customSavingPath);

        // First application of hole filling
        Mat holeFilledImg = Mat::zeros(dialatedImg.size(), CV_8UC1);
        FilteringMeanHeightWidth getHeightWidth = ImageFiltering::getInstance()->applyFloodFill(dialatedImg,
                                                                                                holeFilledImg);
        //dialatedImg.release();
        // BasicAlgo::getInstance()->writeImageGivenPath(holeFilledImg,"/Users/tmondal/Documents/2_hole_filled_Image.jpg");


        IplImage copy = holeFilledImg;
        IplImage *new_image = &copy;
        holeFilledImg = cv::cvarrToMat(ImageFiltering::getInstance()->doImfill(cvCloneImage(new_image)));
        // Second application of hole filling
        customSavingPath = newDirPath +  "/" + "6_hole_filled_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(holeFilledImg, customSavingPath);


        int horThres = ((getHeightWidth.mean_width * 65) / 100);
        holeFilledImg = BasicProcessingTechniques::getInstance()->Horizontal_RLSATechnique(holeFilledImg, horThres);
        customSavingPath = newDirPath +  "/" + "7_rlsa_Image.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(holeFilledImg, customSavingPath);

        // obtaining the foreground image only (one way)
        */
/* Mat foreGdImage = Mat::zeros(holeFilledImg.rows, holeFilledImg.cols, CV_8UC1);
        // foreGdImage.setTo(255);
        BasicAlgo::getInstance()->showImage(foreGdImage);
        bitwise_and(imgGrey,holeFilledImg,foreGdImage);
         *//*


        // obtaining the foreground image only (2nd way)
        Mat foreGdImage = Mat::zeros(holeFilledImg.rows, holeFilledImg.cols, CV_8U);
        foreGdImage.setTo(255);

        for (int iRow = 0; iRow < holeFilledImg.rows; iRow++){
            for (int jCol = 0; jCol < holeFilledImg.cols; jCol++){
                if (holeFilledImg.at<uchar>(iRow, jCol) == 255){
                    foreGdImage.at<uchar>(iRow, jCol) = imgGrey.at<uchar>(iRow, jCol);
                }
            }
        }
        Mat outputBin = Mat::zeros(foreGdImage.rows, foreGdImage.cols, CV_8U);
        PopularDocBinarization::getInstance()->binarizeSafait(foreGdImage,outputBin );
        customSavingPath = newDirPath +  "/" + "8_binarizedImage.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(outputBin,customSavingPath);

        // obtaining the foreground image only
        // Mat foreGdImage;
        Mat clearedForeGdImg = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
        clearedForeGdImg.setTo(255);

        for (int iRow = 0; iRow < clearedForeGdImg.rows; iRow++){
            for (int jCol = 0; jCol < clearedForeGdImg.cols; jCol++){
                if (outputBin.at<uchar>(iRow, jCol) == 0){
                    clearedForeGdImg.at<uchar>(iRow, jCol) = imgGrey.at<uchar>(iRow, jCol);
                }
            }
        }

*/
/*    Mat complementedBinImg = BasicProcessingTechniques::getInstance()->ComplementImage(outputBin);
    bitwise_and(imgGrey,complementedBinImg,clearedForeGdImg);*//*

        customSavingPath = newDirPath +  "/" + "9_onlyForeGdImage.jpg";
        BasicAlgo::getInstance()->writeImageGivenPath(clearedForeGdImg,customSavingPath);


        auto end = chrono::steady_clock::now();
        auto diff = end - start;
        cout << chrono::duration<double, milli>(diff).count() << " ms time taken for execution" << endl;
    }
}


*/

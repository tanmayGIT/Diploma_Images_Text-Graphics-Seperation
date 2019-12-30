//
// Created by tmondal on 14/02/2019.
//

#ifndef CLION_PROJECT_TEXTGRAPHICHSEPERATIONBULK_H
#define CLION_PROJECT_TEXTGRAPHICHSEPERATIONBULK_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include "Binarization/hdr/PopularDocBinarization.hpp"

#include "Binarization/hdr/Feature_Space_Partition_Binarization.h"
#include "Binarization/hdr/GatosBinarization.h"
#include "hdr/DirectoryHandler.hpp"
#include "hdr/AnisotropicGaussianFilter.h"
#include "ImageFiltering/hdr/ImageFiltering.hpp"
#include <opencv2/ximgproc.hpp>
#include "opencv2/core/utility.hpp"
#include <highgui.h>
#include <cv.h>

#include <cstring>
#include <random>
#include <chrono>

using namespace cv::ximgproc;

class TextGraphichSeperationBulk {

public:

    void ApplyBulkEdgeDetection(string allImgDir, string resultSavingDir) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();
        cv::Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(
                "/Users/tmondal/Documents/Workspace_C++/Text_Graphics_Seperation/DocScanImageProcessing/model.yml");


        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.push_back("png");
        validImgExtensions.push_back("jpg");
        validImgExtensions.push_back("bmp");
        validImgExtensions.push_back("tiff");
        validImgExtensions.push_back("tif");
        if (boost::filesystem::exists(allImgDir))
            instance->getFilesInDirectory(allImgDir, validImgFileNames, validImgExtensions);

// #pragma omp parallel
        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string onlyNm = getImgPath.filename().string(); //  getting only the file name with extension
            string keepNmExt = onlyNm;

            string imgFullPath = allImgDir + onlyNm;
            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);

            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }
            Mat imgGrey = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgGrey.channels() >= 3)
                cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);
            int subtractThresh = 0;
            Mat imgGreyCropped = imgGrey(
                    Rect(subtractThresh, subtractThresh, imgGrey.cols - subtractThresh, imgGrey.rows - subtractThresh));
            //   **********************************   Doing Random Forest Based Edge Detection ********************************
            Mat colorImg;
            cv::cvtColor(imgGreyCropped, colorImg, cv::COLOR_GRAY2BGR);
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

            BasicAlgo::getInstance()->writeImageGivenPath(edgeImage, "/Users/tmondal/Documents/Save_Image.png");
            // BasicAlgo::getInstance()->writeImage(edgeImage);
            // BasicAlgo::getInstance()->writeImageGivenPath(edgeImage, "/Users/tmondal/Documents/2_gradient_Image.jpg");

            orientation_map.release();
            edgeImage.release();
            //   **********************************   End of Random Forest Based Edge Detection ********************************



            Mat gray_image = imread("/Users/tmondal/Documents/Save_Image.png", CV_LOAD_IMAGE_GRAYSCALE);

            // *****               Anisotropic diffusion filter                    *****  //
            Mat filteredImage = gray_image.clone();
            filteredImage.convertTo(filteredImage, CV_32FC1);

            AnisotropicGaussianFilter filtImag = *new AnisotropicGaussianFilter();
            filtImag.doAnisotropicDiffusionFiltering(filteredImage, filteredImage.cols, filteredImage.rows);
            double min;
            double max;
            minMaxIdx(filteredImage, &min, &max);
            filteredImage.convertTo(filteredImage, CV_8UC1, 255 / (max - min), -min);
            // BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, "/Users/tmondal/Documents/3_filtered_Image.jpg");
            gray_image.release();


            // *****               Apply Canny Edge Detection                   *****  //
            Mat dummyMat;
            double otsuThreshVal = cv::threshold(filteredImage, dummyMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            dummyMat.release(); // delete the dummy mat
            double highThreshCanny = otsuThreshVal;
            double lowThreshCanny = otsuThreshVal * 0.5;
            Canny(filteredImage, filteredImage, lowThreshCanny, highThreshCanny, 3);
            // BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, "/Users/tmondal/Documents/4_canny_Image.jpg");

            string resultImgSavingName = resultSavingDir + "/" + onlyNm;
            BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, resultImgSavingName);

        }
    }


    void ApplyBulkTextGraphicsSeperation(string allImgDir, string resultSavingDir) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();
        cv::Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection(
                "../model.yml");


        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.push_back("png");
        validImgExtensions.push_back("jpg");
        validImgExtensions.push_back("bmp");
        validImgExtensions.push_back("tiff");
        validImgExtensions.push_back("tif");
        if (boost::filesystem::exists(allImgDir))
            instance->getFilesInDirectory(allImgDir, validImgFileNames, validImgExtensions);


// #pragma omp parallel
        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string onlyNm = getImgPath.filename().string(); //  getting only the file name with extension
            string keepNmExt = onlyNm;

            string getOnlyExt = getImgPath.extension().string();
            onlyNm.erase(onlyNm.find_last_of("."), string::npos);

            string imgFullPath = allImgDir + keepNmExt;
            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);

            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }
            Mat imgGrey = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgGrey.channels() >= 3)
                cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);
            int subtractThresh = 0;
            Mat imgGreyCropped = imgGrey(
                    Rect(subtractThresh, subtractThresh, imgGrey.cols - subtractThresh, imgGrey.rows - subtractThresh));
            //   **********************************   Doing Random Forest Based Edge Detection ********************************
            Mat colorImg;
            cv::cvtColor(imgGreyCropped, colorImg, cv::COLOR_GRAY2BGR);
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

            BasicAlgo::getInstance()->writeImageGivenPath(edgeImage, "/home/mondal/Documents/Save_Image.png");
            //   BasicAlgo::getInstance()->writeImage(edgeImage);
            // BasicAlgo::getInstance()->writeImageGivenPath(edgeImage, "/home/mondal/Documents/2_gradient_Image.jpg");

            orientation_map.release();
            edgeImage.release();
            //   **********************************   End of Random Forest Based Edge Detection ********************************



            Mat gray_image = imread("/home/mondal/Documents/Save_Image.png", CV_LOAD_IMAGE_GRAYSCALE);

            // *****               Anisotropic diffusion filter                    *****  //
            Mat filteredImage = gray_image.clone();
            filteredImage.convertTo(filteredImage, CV_32FC1);

            AnisotropicGaussianFilter filtImag = *new AnisotropicGaussianFilter();
            filtImag.doAnisotropicDiffusionFiltering(filteredImage, filteredImage.cols, filteredImage.rows);
            double min;
            double max;
            minMaxIdx(filteredImage, &min, &max);
            filteredImage.convertTo(filteredImage, CV_8UC1, 255 / (max - min), -min);
           //  BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, "/home/mondal/Documents/3_filtered_Image.jpg");
            gray_image.release();


            // *****               Apply Canny Edge Detection                   *****  //
            Mat dummyMat;
            double otsuThreshVal = cv::threshold(filteredImage, dummyMat, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
            dummyMat.release(); // delete the dummy mat
            double highThreshCanny = otsuThreshVal;
            double lowThreshCanny = otsuThreshVal * 0.5;
            Canny(filteredImage, filteredImage, lowThreshCanny, highThreshCanny, 3);

            string customSavingPathCanny = resultSavingDir + "/" + "Canny_" + onlyNm + ".png";
            BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, customSavingPathCanny);
          //   BasicAlgo::getInstance()->writeImageGivenPath(filteredImage, "/home/mondal/Documents/4_canny_Image.jpg");


            // Perform dilation operation
            Mat dialationSE = getStructuringElement(MORPH_RECT, Size(7, 7));   // Define the structuring elements
            Mat dialatedImg;
            morphologyEx(filteredImage, dialatedImg, MORPH_DILATE, dialationSE); // Perform dilation operation
            filteredImage.release();
           //  BasicAlgo::getInstance()->writeImageGivenPath(dialatedImg,"/home/mondal/Documents/5_canny_dialted_Image.jpg");
            Mat holeFilledImg = Mat::zeros(dialatedImg.size(), CV_8U);


            Mat copyDialedImg = Mat::zeros(dialatedImg.size(), CV_8U); // remove the border problems as the border was there and due to that when filling was done , everything inside the border were getting filled
            int xStart  = 0; int yStart = 0;
            dialatedImg(Rect(xStart, yStart, (dialatedImg.cols-xStart), (dialatedImg.rows-yStart) )).
                    copyTo(copyDialedImg(Rect(xStart, yStart, (dialatedImg.cols-xStart), (dialatedImg.rows-yStart) )));
            // BasicAlgo::getInstance()->writeImageGivenPath(copyDialedImg,"/home/mondal/Documents/croppedDilaed_Image.jpg");

            // First application of hole filling
            FilteringMeanHeightWidth getHeightWidth = ImageFiltering::getInstance()->applyFloodFill(copyDialedImg,
                                                                                                    holeFilledImg);
            dialatedImg.release();
            // BasicAlgo::getInstance()->writeImageGivenPath(holeFilledImg,"/home/mondal/Documents/6_hole_filled_Image.jpg");


            // obtaining the foreground image only (2nd way)
            Mat foreGdImage = Mat::zeros(holeFilledImg.rows, holeFilledImg.cols, CV_8U);
            foreGdImage.setTo(255);


            for (int iRow = 0; iRow < holeFilledImg.rows; iRow++) {
                for (int jCol = 0; jCol < holeFilledImg.cols; jCol++) {
                    if (holeFilledImg.at<uchar>(iRow, jCol) == 255) {
                        foreGdImage.at<uchar>(iRow, jCol) = imgGreyCropped.at<uchar>(iRow, jCol);
                    }
                }
            }
           // BasicAlgo::getInstance()->writeImageGivenPath(foreGdImage, "/home/mondal/Documents/12_ForeGD_Image.jpg");

            // obtaining the foreground image only (2nd way)
            Mat fullForeGdImage = Mat::zeros(imgGrey.rows, imgGrey.cols, CV_8U);
            fullForeGdImage.setTo(255);
            foreGdImage.copyTo(fullForeGdImage(Rect(subtractThresh, subtractThresh, imgGrey.cols - subtractThresh,
                                                    imgGrey.rows - subtractThresh)));

            string customSavingPath = resultSavingDir + "/" + onlyNm + ".png";
            // BasicAlgo::getInstance()->showImage(fullForeGdImage);
            BasicAlgo::getInstance()->writeImageGivenPath(fullForeGdImage, customSavingPath);
        }

    }


    /*
 * Here we are trying to use the canny edge detected image as SSP nad use this SSP image to calculate the feature image of Savoula and Niblack
 */
    void ApplyFinalTextGraphicsRemoval_2(string allFileImgNm, string allImgDir, string resultSavingDir,
                                         string cannyImageSavingPath) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();

        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.emplace_back("png");
        validImgExtensions.emplace_back("jpg");
        validImgExtensions.emplace_back("bmp");
        validImgExtensions.emplace_back("tiff");
        if (boost::filesystem::exists(allFileImgNm))
            instance->getFilesInDirectory(allFileImgNm, validImgFileNames, validImgExtensions);

        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string onlyNm = getImgPath.filename().string(); //  getting only the file name
            string keepNmExt = onlyNm;

            string getOnlyExt = getImgPath.extension().string();
            onlyNm.erase(onlyNm.find_last_of("."), string::npos);
            string imgFullPath = allFileImgNm + keepNmExt;
            string cannyImagFullPath = cannyImageSavingPath + "Canny_" + onlyNm + ".png";

            Mat cannyImag = imread(cannyImagFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);
            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);

            if (cannyImag.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            Mat imgGrey = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgGrey.channels() >= 3)
                cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);


            int strokeWidth = PopularDocBinarization::getInstance()->CalculateStrokeWidth(imgGrey, cannyImag);

            Mat calculatedFeatureImage = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
            PopularDocBinarization::getInstance()->callBinarizationFeature(imgGrey, calculatedFeatureImage, cannyImag, strokeWidth, strokeWidth, 'n'); // Niblack, Savoula, Wolf_Jolin


            string sureConfusedImgName = allImgDir + "/" + onlyNm + "_Sure_and_Confused_Text_Imag" + ".png";
            string sureImagName = allImgDir + "/" + onlyNm + "_Sure_Text_Imag" + ".png";
            string textFileName = allImgDir + "/" + onlyNm + "_TextFile" + ".txt";

            // Now read the  "Sure and Confused Text Segmentation" and "Sure Text Result" images and the text file containing line segmentation results
            Mat sureTextImage = imread(sureImagName, CV_LOAD_IMAGE_UNCHANGED);
            Mat sureConfusedTextImage = imread(sureConfusedImgName, CV_LOAD_IMAGE_UNCHANGED);


            // Read the file
            std::fstream myfile(textFileName);
            int avgLineHeight, startLineRow, endLineRow;
            myfile >> avgLineHeight >> startLineRow >> endLineRow;


            int winy = std::round(strokeWidth/2), winx = std::round(strokeWidth/2);

            // initializing with the sure text pixel image and then we will addd more pixels by additional checking
            Mat CombineBinImage = sureTextImage;// Mat(imgGrey.size(), imgGrey.type(), Scalar(255));
            Mat anotherImageFinal =  ApplyNeighbourhoodCorrectionSymmetric_1(imgGrey, sureConfusedTextImage, calculatedFeatureImage
                                                                                    , CombineBinImage, cannyImag, winy, winx);
/*            ApplyNeighbourhoodCorrection(imgGrey, sureConfusedTextImage, BinaryWholeImage_Savoula,
                                         DynamicWindowBinImage, CombineBinImage, winy, winx );*/

            string ResultSavingPath = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BG_Seperated/Final_Result-6/";

            int morph_size = 2;
            Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                Point(morph_size, morph_size));
            Mat openedImage1 = CombineBinImage; // result matrix
            // morphologyEx(CombineBinImage, openedImage, MORPH_OPEN, element, Point(-1, -1), 1);
            cv::threshold(openedImage1, openedImage1, 0, 255, CV_THRESH_OTSU);

            Mat openedImage2 = anotherImageFinal; // result matrix
            // morphologyEx(CombineBinImage, openedImage, MORPH_OPEN, element, Point(-1, -1), 1);
            cv::threshold(openedImage2, openedImage2, 0, 255, CV_THRESH_OTSU);



            //  BasicAlgo::getInstance()->writeImageGivenPath(CombineBinImage,"/Users/tmondal/Documents/afterOpingCombined_Binary_Image.jpg");
            string resultImgSavingName = resultSavingDir + onlyNm + "Output" + ".png";
            string resultImgSavingName_1 = ResultSavingPath + onlyNm + "Output" + ".png";

            BasicAlgo::getInstance()->writeImageGivenPath(openedImage1, resultImgSavingName); // reconstruct using the cluster in complete
            BasicAlgo::getInstance()->writeImageGivenPath(openedImage2, resultImgSavingName_1); // pixelwise reconstruction
        }
    }


    /*
     * Here we are trying to use the canny edge detected image as SSP and then instead of labeling each pixels separately,
     * we propose to modify the cluster of pixels
     */
    void ApplyFinalTextGraphicsRemoval_1(string allFileImgNm, string allImgDir, string resultSavingDir,
                                         string cannyImageSavingPath) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();

        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.push_back("png");
        validImgExtensions.push_back("jpg");
        validImgExtensions.push_back("bmp");
        validImgExtensions.push_back("tiff");
        if (boost::filesystem::exists(allFileImgNm))
            instance->getFilesInDirectory(allFileImgNm, validImgFileNames, validImgExtensions);

        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string onlyNm = getImgPath.filename().string(); //  getting only the file name
            string keepNmExt = onlyNm;

            //cout << keepNmExt << endl;

            string getOnlyExt = getImgPath.extension().string();
            onlyNm.erase(onlyNm.find_last_of("."), string::npos);
            string imgFullPath = allFileImgNm + keepNmExt;
            string cannyImagFullPath = cannyImageSavingPath + "Canny_" + onlyNm + ".png";

            Mat cannyImag = imread(cannyImagFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);
            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);

            if (cannyImag.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            Mat imgGrey = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgGrey.channels() >= 3)
                cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);


            int strokeWidth = PopularDocBinarization::getInstance()->CalculateStrokeWidth(imgGrey, cannyImag);

            Mat calculatedFeatureImage = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
        //    PopularDocBinarization::getInstance()->callBinarizationFeature(imgGrey, calculatedFeatureImage, cannyImag, strokeWidth, strokeWidth, 's'); // Niblack, Savoula, Wolf_Jolin





            string sureConfusedImgName = allImgDir + "/" + onlyNm + "_Sure_and_Confused_Text_Imag" + ".png";
            string sureImagName = allImgDir + "/" + onlyNm + "_Sure_Text_Imag" + ".png";
            string textFileName = allImgDir + "/" + onlyNm + "_TextFile" + ".txt";

            // Now read the  "Sure and Confused Text Segmentation" and "Sure Text Result" images and the text file containing line segmentation results
            Mat sureTextImage = imread(sureImagName, CV_LOAD_IMAGE_UNCHANGED);
            Mat sureConfusedTextImage = imread(sureConfusedImgName, CV_LOAD_IMAGE_UNCHANGED);

            Mat BinaryWholeImage_Savoula = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
            PopularDocBinarization::getInstance()->callBinarizationFunction(imgGrey, BinaryWholeImage_Savoula,
                                                                            's'); // Niblack, Savoula, Wolf_Jolin

            //  BasicAlgo::getInstance()->writeImageGivenPath(imgGrey,"/Users/tmondal/Documents/originalTobeBinarized_Image.jpg");
            //  BasicAlgo::getInstance()->writeImageGivenPath(BinaryWholeImage_Savoula,"/Users/tmondal/Documents/savoula_Image.jpg");


            // Read the file
            std::fstream myfile(textFileName);
            int avgLineHeight, startLineRow, endLineRow;
            myfile >> avgLineHeight >> startLineRow >> endLineRow;
            // printf("%d\t %d \t %d \n",avgLineHeight, startLineRow, endLineRow);
            vector<blockImagProperties> keepAllBlockImag;
            PopularDocBinarization::getInstance()->LineWiseDivideImageIntoSubImages(imgGrey, avgLineHeight,
                                                                                    imgGrey.cols, startLineRow,
                                                                                    endLineRow, keepAllBlockImag);

            Mat DynamicWindowBinImage = (Mat::zeros(imgGrey.size(), imgGrey.type()));
            DynamicWindowBinImage = BasicProcessingTechniques::getInstance()->ComplementImage(
                    DynamicWindowBinImage.clone());
            for (int iiSubParts = 0; iiSubParts < keepAllBlockImag.size(); iiSubParts++) {
                cv::Rect getROI;

                getROI.x = keepAllBlockImag.at(iiSubParts).startX;
                getROI.width = keepAllBlockImag.at(iiSubParts).endX;
                getROI.y = keepAllBlockImag.at(iiSubParts).startY;
                getROI.height = keepAllBlockImag.at(iiSubParts).endY;

                Mat getSubSubImag = imgGrey(getROI);
                Mat getSubSubBinaryImag = sureTextImage(getROI);

                Mat resultBinSubImage;
                PopularDocBinarization::getInstance()->SimpleDynamicWindowBased_Binarization(getSubSubImag,
                                                                                             getSubSubBinaryImag,
                                                                                             resultBinSubImage);
                resultBinSubImage.copyTo(DynamicWindowBinImage(getROI));
            }
            //  BasicAlgo::getInstance()->writeImageGivenPath(DynamicWindowBinImage,"/Users/tmondal/Documents/dynamicBoundary_Image.jpg");


            // int winy = 5, winx = 5; // for 5x5 window check
            int winy = std::round(strokeWidth/2), winx = std::round(strokeWidth/2);

            // initializing with the sure text pixel image and then we will addd more pixels by additional checking
            Mat CombineBinImage = sureTextImage;// Mat(imgGrey.size(), imgGrey.type(), Scalar(255));

            Mat anotherImageFinal =  ApplyNeighbourhoodCorrectionSymmetric(imgGrey, sureConfusedTextImage, BinaryWholeImage_Savoula,
                                                  DynamicWindowBinImage, CombineBinImage, cannyImag, winy, winx);



/*            ApplyNeighbourhoodCorrection(imgGrey, sureConfusedTextImage, BinaryWholeImage_Savoula,
                                         DynamicWindowBinImage, CombineBinImage, winy, winx );*/

            string ResultSavingPath = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BG_Seperated/No-Filled-Contours/Final_Result-4/";

            int morph_size = 2;
            Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                Point(morph_size, morph_size));
            Mat openedImage_1 = CombineBinImage; // result matrix
            Mat openedImage_2 = anotherImageFinal; // result matrix
            // morphologyEx(CombineBinImage, openedImage, MORPH_OPEN, element, Point(-1, -1), 1);
            cv::threshold(openedImage_1, openedImage_1, 0, 255, CV_THRESH_OTSU);
            cv::threshold(openedImage_2, openedImage_2, 0, 255, CV_THRESH_OTSU);


            //  BasicAlgo::getInstance()->writeImageGivenPath(CombineBinImage,"/Users/tmondal/Documents/afterOpingCombined_Binary_Image.jpg");
            string resultImgSavingName = resultSavingDir + onlyNm + "Output" + ".png";
            string resultImgSavingName_1 = ResultSavingPath + onlyNm + "Output" + ".png";

            BasicAlgo::getInstance()->writeImageGivenPath(openedImage_1, resultImgSavingName);
            BasicAlgo::getInstance()->writeImageGivenPath(openedImage_2, resultImgSavingName_1);
        }
    }

    void ApplyClassicBinarizationTechniques(string allFileImgNm, string resultSavingDir) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();

        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.push_back("png");
        validImgExtensions.push_back("jpg");
        validImgExtensions.push_back("bmp");
        validImgExtensions.push_back("tiff");
        validImgExtensions.push_back("tif");
        if (boost::filesystem::exists(allFileImgNm))
            instance->getFilesInDirectory(allFileImgNm, validImgFileNames, validImgExtensions);

        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string onlyNm = getImgPath.filename().string(); //  getting only the file name
            string keepNmExt = onlyNm;

            // cout << keepNmExt << endl;

            string getOnlyExt = getImgPath.extension().string();
            onlyNm.erase(onlyNm.find_last_of("."), string::npos);
            string imgFullPath = allFileImgNm + keepNmExt;


            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);
            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            Mat imgGrey = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgGrey.channels() >= 3)
                cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);

            Mat BinaryWholeImage_Savoula = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
            Mat BinaryWholeImage_Niblack = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
            Mat BinaryWholeImage_WolfJolin = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());

            PopularDocBinarization::getInstance()->callBinarizationFunction(imgGrey, BinaryWholeImage_Savoula,'s'); // Savoula
            PopularDocBinarization::getInstance()->callBinarizationFunction(imgGrey, BinaryWholeImage_Niblack,'n'); // Niblack
            PopularDocBinarization::getInstance()->callBinarizationFunction(imgGrey, BinaryWholeImage_WolfJolin,'w'); // Wolf_Jolin

            string resultImgSavingNameSavoula = resultSavingDir + "savoula" + "/" + onlyNm + ".png";
            string resultImgSavingNameNiblack = resultSavingDir + "niblack" + "/" + onlyNm + ".png";
            string resultImgSavingNameWolfJolin = resultSavingDir + "wolfjolin" + "/" + onlyNm + ".png";

            BasicAlgo::getInstance()->writeImageGivenPath(BinaryWholeImage_Savoula, resultImgSavingNameSavoula);
            BasicAlgo::getInstance()->writeImageGivenPath(BinaryWholeImage_Niblack, resultImgSavingNameNiblack);
            BasicAlgo::getInstance()->writeImageGivenPath(BinaryWholeImage_WolfJolin, resultImgSavingNameWolfJolin);
        }
    }


    void ApplyFinalTextGraphicsRemoval(string allFileImgNm, string allImgDir, string resultSavingDir,
            string cannyImageSavingPath ) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();

        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.push_back("png");
        validImgExtensions.push_back("jpg");
        validImgExtensions.push_back("bmp");
        validImgExtensions.push_back("tiff");
        validImgExtensions.push_back("tif");
        if (boost::filesystem::exists(allFileImgNm))
            instance->getFilesInDirectory(allFileImgNm, validImgFileNames, validImgExtensions);

        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string onlyNm = getImgPath.filename().string(); //  getting only the file name
            string keepNmExt = onlyNm;

            // cout << keepNmExt << endl;

            string getOnlyExt = getImgPath.extension().string();
            onlyNm.erase(onlyNm.find_last_of("."), string::npos);
            string imgFullPath = allFileImgNm + keepNmExt;

            string cannyImagFullPath = cannyImageSavingPath + "Canny_" + onlyNm + ".png";
            Mat cannyImag = imread(cannyImagFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);

            if (cannyImag.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);
            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }

            Mat imgGrey = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgGrey.channels() >= 3)
                cv::cvtColor(imgGrey, imgGrey, CV_BGR2GRAY);

            int strokeWidth = PopularDocBinarization::getInstance()->CalculateStrokeWidth(imgGrey, cannyImag);

            string sureConfusedImgName = allImgDir + "/" + onlyNm + "_Sure_and_Confused_Text_Imag" + ".png";
            string sureImagName = allImgDir + "/" + onlyNm + "_Sure_Text_Imag" + ".png";
            string textFileName = allImgDir + "/" + onlyNm + "_TextFile" + ".txt";

            // Now read the  "Sure and Confused Text Segmentation" and "Sure Text Result" images and the text file containing line segmentation results
            Mat sureTextImage = imread(sureImagName, CV_LOAD_IMAGE_UNCHANGED);
            Mat sureConfusedTextImage = imread(sureConfusedImgName, CV_LOAD_IMAGE_UNCHANGED);
            Mat BinaryWholeImage_Savoula = Mat::zeros(imgGrey.rows, imgGrey.cols, imgGrey.type());
            PopularDocBinarization::getInstance()->callBinarizationFunction(imgGrey, BinaryWholeImage_Savoula,
                                                                            's'); // Niblack, Savoula, Wolf_Jolin

            //  BasicAlgo::getInstance()->writeImageGivenPath(imgGrey,"/Users/tmondal/Documents/originalTobeBinarized_Image.jpg");
            //  BasicAlgo::getInstance()->writeImageGivenPath(BinaryWholeImage_Savoula,"/Users/tmondal/Documents/savoula_Image.jpg");



            // Read the file
            std::fstream myfile(textFileName);
            int avgLineHeight, startLineRow, endLineRow;
            myfile >> avgLineHeight >> startLineRow >> endLineRow;
            // printf("%d\t %d \t %d \n",avgLineHeight, startLineRow, endLineRow);
            vector<blockImagProperties> keepAllBlockImag;
            PopularDocBinarization::getInstance()->LineWiseDivideImageIntoSubImages(imgGrey, avgLineHeight,
                                                                                    imgGrey.cols, startLineRow,
                                                                                    endLineRow, keepAllBlockImag);

            Mat DynamicWindowBinImage = (Mat::zeros(imgGrey.size(), imgGrey.type()));
            DynamicWindowBinImage = BasicProcessingTechniques::getInstance()->ComplementImage(
                    DynamicWindowBinImage.clone());
            for (int iiSubParts = 0; iiSubParts < keepAllBlockImag.size(); iiSubParts++) {
                cv::Rect getROI;

                getROI.x = keepAllBlockImag.at(iiSubParts).startX;
                getROI.width = keepAllBlockImag.at(iiSubParts).endX;
                getROI.y = keepAllBlockImag.at(iiSubParts).startY;
                getROI.height = keepAllBlockImag.at(iiSubParts).endY;

                Mat getSubSubImag = imgGrey(getROI);
                Mat getSubSubBinaryImag = sureTextImage(getROI);

                Mat resultBinSubImage;
                PopularDocBinarization::getInstance()->SimpleDynamicWindowBased_Binarization(getSubSubImag,
                                                                                             getSubSubBinaryImag,
                                                                                             resultBinSubImage);
                resultBinSubImage.copyTo(DynamicWindowBinImage(getROI));
            }
            //  BasicAlgo::getInstance()->writeImageGivenPath(DynamicWindowBinImage,"/Users/tmondal/Documents/dynamicBoundary_Image.jpg");


            int winy = 5, winx = 5; // for 5x5 window check
            // int winy = std::round(strokeWidth/2), winx = std::round(strokeWidth/2);


            Mat CombineBinImage = sureTextImage;// Mat(imgGrey.size(), imgGrey.type(), Scalar(255));
            ApplyNeighbourhoodCorrection(imgGrey, sureConfusedTextImage, BinaryWholeImage_Savoula,
                                         DynamicWindowBinImage, CombineBinImage, winy, winx);


            int morph_size = 2;
            Mat element = getStructuringElement(MORPH_RECT, Size(2 * morph_size + 1, 2 * morph_size + 1),
                                                Point(morph_size, morph_size));
            Mat openedImage = CombineBinImage; // result matrix
            // morphologyEx(CombineBinImage, openedImage, MORPH_OPEN, element, Point(-1, -1), 1);
            cv::threshold(openedImage, openedImage, 0, 255, CV_THRESH_OTSU);
            //  BasicAlgo::getInstance()->writeImageGivenPath(CombineBinImage,"/Users/tmondal/Documents/afterOpingCombined_Binary_Image.jpg");
            string resultImgSavingName = resultSavingDir + "/" + onlyNm + ".png";
            BasicAlgo::getInstance()->writeImageGivenPath(openedImage, resultImgSavingName);
        }
    }





    Mat ApplyNeighbourhoodCorrectionSymmetric_1(Mat &imgGrey, Mat &sureConfusedTextImage, Mat &featureImage,
                                                Mat &CombineBinImage, Mat &cannySSPImag, int winy, int winx)
    {

/*       BasicAlgo::getInstance()->showImage(imgGrey);
       BasicAlgo::getInstance()->showImage(sureConfusedTextImage);
       BasicAlgo::getInstance()->showImage(featureImage);
       BasicAlgo::getInstance()->showImage(CombineBinImage);
       BasicAlgo::getInstance()->showImage(cannySSPImag);*/


        float alphaCut = 0.2;
        Mat sureConfusedTextImageCopy = CombineBinImage;

        // Get unique values in the Mat to understand the number of clusters actually obtained
        std::vector<float> getAllUniqueVals = BasicAlgo::getInstance()->uniqueValuesInMat(sureConfusedTextImage,true);

        Mat clusterWiseImage = Mat::zeros(sureConfusedTextImage.size(), sureConfusedTextImage.type());
        for (int iRow = 0; iRow < imgGrey.rows; iRow++) {
            for (int jCol = 0; jCol < imgGrey.cols; jCol++) {
                if ((sureConfusedTextImage.at<uchar>(iRow, jCol) != 0) &&
                    (sureConfusedTextImage.at<uchar>(iRow, jCol) != 255)) { // the other value
                    clusterWiseImage.at<uchar>(iRow, jCol) = 255;
                }
            }
        }
        //  BasicAlgo::getInstance()->showImage(clusterWiseImage);
        // Now I need to do connected component labelling which will return the number of pixels in each CC and also the total number of CCs

        vector<ComponentInfo> cominfoForBinImgs; // here we are getting all the info of each CC i.e. xStart, yStart, Width, Height, centroid and area
        int avgHeightOrig, avgWidthOrig;
        vector<bool> labelCompHeirarchyOrig;
        // you need it here just to calculate the average width of CCs
        BasicProcessingTechniques::getInstance()->getConnectedCompOperationPPPImage(clusterWiseImage, cominfoForBinImgs,
                                                                                    labelCompHeirarchyOrig,
                                                                                    avgHeightOrig, avgWidthOrig);
        labelCompHeirarchyOrig.clear();
        // Mat chkImag = Mat::zeros(clusterWiseImage.size(), clusterWiseImage.type());
        for (int iComp = 0; iComp < cominfoForBinImgs.size(); iComp++) {
            int compGoodPixelCnt = 0;
            if(cominfoForBinImgs[iComp].area > 0){
                int sizeOfComp = cominfoForBinImgs[iComp].outerBlobPts.size();
                for (int iPts = 0; iPts < sizeOfComp; iPts++) {
                    int getX =  cominfoForBinImgs[iComp].outerBlobPts[iPts].x;
                    int getY = cominfoForBinImgs[iComp].outerBlobPts[iPts].y;

                    int sumNeighBorThreshCnt = CheckThresholdCondition_1(winx, winy, getY, getX, cannySSPImag, featureImage,imgGrey );
                    // chkImag.at<uchar>(getY, getX) = 255;
                    if(sumNeighBorThreshCnt > 0) {
                        sureConfusedTextImageCopy.at<uchar>(getY, getX) = 255;
                        compGoodPixelCnt++;
                    }
                }

                int checkLimitCmp = round(alphaCut * sizeOfComp);
                if(checkLimitCmp < 1)
                    checkLimitCmp = 1;

                if(compGoodPixelCnt > checkLimitCmp){
                    for (int iPts = 0; iPts < sizeOfComp; iPts++) {
                        int getX = cominfoForBinImgs[iComp].outerBlobPts[iPts].x;
                        int getY = cominfoForBinImgs[iComp].outerBlobPts[iPts].y;

                        CombineBinImage.at<uchar>(getY, getX) = 255; // make it foreground /text pixel
                    }
                }
            }
        }
        return sureConfusedTextImageCopy;
    }


    Mat ApplyNeighbourhoodCorrectionSymmetric(Mat &imgGrey, Mat &sureConfusedTextImage, Mat &BinaryWholeImage_Savoula,
                                               Mat &DynamicWindowBinImage, Mat &CombineBinImage, Mat &cannySSPImag,
                                               int winy, int winx)
    {

        float alphaCut = 0.2;
        Mat sureConfusedTextImageCopy = sureConfusedTextImage;

        // Get unique values in the Mat to understand the number of clusters actually obtained
        std::vector<float> getAllUniqueVals = BasicAlgo::getInstance()->uniqueValuesInMat(sureConfusedTextImage,true);

        Mat clusterWiseImage = Mat::zeros(sureConfusedTextImage.size(), sureConfusedTextImage.type());
        for (int iRow = 0; iRow < imgGrey.rows; iRow++) {
            for (int jCol = 0; jCol < imgGrey.cols; jCol++) {
                if ((sureConfusedTextImage.at<uchar>(iRow, jCol) != 0) &&
                    (sureConfusedTextImage.at<uchar>(iRow, jCol) != 255)) { // the other value
                    clusterWiseImage.at<uchar>(iRow, jCol) = 255;
                }
            }
        }
      //  BasicAlgo::getInstance()->showImage(clusterWiseImage);
        // Now I need to do connected component labelling which will return the number of pixels in each CC and also the total number of CCs

        vector<ComponentInfo> cominfoForBinImgs; // here we are getting all the info of each CC i.e. xStart, yStart, Width, Height, centroid and area
        int avgHeightOrig, avgWidthOrig;
        vector<bool> labelCompHeirarchyOrig;
        // you need it here just to calculate the average width of CCs
        BasicProcessingTechniques::getInstance()->getConnectedCompOperationPPPImage(clusterWiseImage, cominfoForBinImgs,
                                                                                    labelCompHeirarchyOrig,
                                                                                    avgHeightOrig, avgWidthOrig);
        labelCompHeirarchyOrig.clear();
       // Mat chkImag = Mat::zeros(clusterWiseImage.size(), clusterWiseImage.type());
        for (int iComp = 0; iComp < cominfoForBinImgs.size(); iComp++) {
            int compGoodPixelCnt = 0;
            if(cominfoForBinImgs[iComp].area > 0){
                int sizeOfComp = cominfoForBinImgs[iComp].outerBlobPts.size();
                for (int iPts = 0; iPts < sizeOfComp; iPts++) {
                   int getX =  cominfoForBinImgs[iComp].outerBlobPts[iPts].x;
                   int getY = cominfoForBinImgs[iComp].outerBlobPts[iPts].y;

                    int sumNeighBorThreshCnt = CheckThresholdCondition(winx, winy, getY, getX, cannySSPImag );
                  // chkImag.at<uchar>(getY, getX) = 255;
                //   int sumNeighBorThreshCnt = CheckThresholdCondition_BinaryCheck(winx, winy, getY, getX, cannySSPImag, BinaryWholeImage_Savoula, DynamicWindowBinImage );
                   if(sumNeighBorThreshCnt > 0) {
                       sureConfusedTextImageCopy.at<uchar>(getY, getX) = 255;
                       compGoodPixelCnt++;
                   }
                }

                int checkLimitCmp = round(alphaCut * sizeOfComp);
                if(checkLimitCmp < 1)
                    checkLimitCmp = 1;

                if(compGoodPixelCnt > checkLimitCmp){
                    for (int iPts = 0; iPts < sizeOfComp; iPts++) {
                        int getX = cominfoForBinImgs[iComp].outerBlobPts[iPts].x;
                        int getY = cominfoForBinImgs[iComp].outerBlobPts[iPts].y;

                        CombineBinImage.at<uchar>(getY, getX) = 255; // make it foreground /text pixel
                    }
                }
            }

        }
  //  BasicAlgo::getInstance()->writeImage(chkImag);
    return sureConfusedTextImageCopy;
    }


    void ApplyNeighbourhoodCorrection(Mat &imgGrey, Mat &sureConfusedTextImage, Mat &BinaryWholeImage_Savoula,
                                      Mat &DynamicWindowBinImage, Mat &CombineBinImage, int winy, int winx) {

/*    BasicAlgo::getInstance()->showImage(CombineBinImage);
    BasicAlgo::getInstance()->showImage(sureConfusedTextImage);*/

        for (int iRow = 0; iRow < imgGrey.rows; iRow++) {
            for (int jCol = 0; jCol < imgGrey.cols; jCol++) {

                if ((sureConfusedTextImage.at<uchar>(iRow, jCol) != 0) &&
                    (sureConfusedTextImage.at<uchar>(iRow, jCol) != 255)) { // the other value
                    // check if this value is present in both in savoula and dynamic window image, whether they both are text pixels or not
                    if ((BinaryWholeImage_Savoula.at<uchar>(iRow, jCol) == 0) &&
                        (DynamicWindowBinImage.at<uchar>(iRow, jCol) == 0)) {

                        // sureConfusedTextImage.at<uchar>(iRow, jCol) = 0;
                        // CombineBinImage.at<uchar>(iRow, jCol) = 0;
                        // see in the neighbourhood
                        int s_cnt = 0;
                        int x1 = 0, y1 = 0;
                        for (int k = -(winy / 2); k <= (winy / 2); k++) {
                            for (int j = -(winx / 2); j <= (winx / 2); j++) {
                                if ((k != 0) && (j != 0)) { // to avoid the center pixel
                                    x1 = PopularDocBinarization::getInstance()->reflect(imgGrey.cols, jCol - j);
                                    y1 = PopularDocBinarization::getInstance()->reflect(imgGrey.rows, iRow - k);
                                    if ((BinaryWholeImage_Savoula.at<uchar>(y1, x1) == 0) && (DynamicWindowBinImage.at<uchar>(y1, x1) == 0))
                                   // if (sureConfusedTextImage.at<uchar>(y1, x1) == 0) // counting the number of strong pixels at the surrounding
                                        s_cnt++;
                                }
                            }
                        }

                        if (s_cnt > (0.2 * ((winy * winx) -
                                            1))) { // if number of strong pixels is more that 40% of total pixels in the neighbourghood (-1 to avoid the center pixel)
                            CombineBinImage.at<uchar>(iRow, jCol) = 0;
                            sureConfusedTextImage.at<uchar>(iRow, jCol) = 0;
                        }


                    }
                }
            }
        }

    }


    inline int CheckThresholdCondition_1(int winX, int winY, int iiToSee, int jjToSee, Mat& cannySSPImag, Mat& featureImag,  Mat& greyImag ){

        // Calculate the number of strong/SSP pixels around this confused pixel of the component
        int sumNeighBorThreshCnt = 0;
        for (int k = -winY; k <= winY; k++){
            for (int j = -winX; j <= winX; j++){
                if ( (k != 0) && (j != 0) ){
                    int iiLimit = (iiToSee + k);
                    if(iiLimit < 0)
                        iiLimit = 0;
                    else if (iiLimit >= cannySSPImag.rows)
                        iiLimit = cannySSPImag.rows - 1;

                    int jjLimit = (jjToSee + j);
                    if(jjLimit < 0)
                        jjLimit = 0;
                    else if (jjLimit >= cannySSPImag.cols)
                        jjLimit = cannySSPImag.cols - 1;

                    if( cannySSPImag.at<uchar>(iiLimit, jjLimit) == 255 ){
                        float getLocalThresh = featureImag.at<uchar>(iiLimit, jjLimit);
                        if(greyImag.at<uchar>(iiToSee, jjToSee) <=  getLocalThresh )
                            sumNeighBorThreshCnt++;
                        else
                            sumNeighBorThreshCnt--;
                    }
                }
            }
        }
        return sumNeighBorThreshCnt;
    }
    inline int CheckThresholdCondition(int winX, int winY, int iiToSee, int jjToSee, Mat& cannySSPImag ){

        // Calculate the number of strong/SSP pixels around this confused pixel of the component
        int sumNeighBorThreshCnt = 0;
        for (int k = -winY; k <= winY; k++){
            for (int j = -winX; j <= winX; j++){
                if ( (k != 0) && (j != 0) ){
                    int iiLimit = (iiToSee + k);
                    if(iiLimit < 0)
                        iiLimit = 0;
                    else if (iiLimit >= cannySSPImag.rows)
                        iiLimit = cannySSPImag.rows - 1;

                    int jjLimit = (jjToSee + j);
                    if(jjLimit < 0)
                        jjLimit = 0;
                    else if (jjLimit >= cannySSPImag.cols)
                        jjLimit = cannySSPImag.cols - 1;

                    if( cannySSPImag.at<uchar>(iiLimit, jjLimit) == 255 ){
                        sumNeighBorThreshCnt++;
                    }
                }
            }
        }
        return sumNeighBorThreshCnt;
    }

    inline int CheckThresholdCondition_BinaryCheck(int winX, int winY, int iiToSee, int jjToSee, Mat& cannySSPImag, Mat &BinaryWholeImage_Savoula,
                                                   Mat &DynamicWindowBinImage ){

        // Calculate the number of strong/SSP pixels around this confused pixel of the component
        int sumNeighBorThreshCnt = 0;
        for (int k = -winY; k <= winY; k++){
            for (int j = -winX; j <= winX; j++){
                if ( (k != 0) && (j != 0) ){
                    int iiLimit = (iiToSee + k);
                    if(iiLimit < 0)
                        iiLimit = 0;
                    else if (iiLimit >= cannySSPImag.rows)
                        iiLimit = cannySSPImag.rows - 1;

                    int jjLimit = (jjToSee + j);
                    if(jjLimit < 0)
                        jjLimit = 0;
                    else if (jjLimit >= cannySSPImag.cols)
                        jjLimit = cannySSPImag.cols - 1;

                    if( cannySSPImag.at<uchar>(iiLimit, jjLimit) == 255 ){ // if there is a SSP pixel
                        if( (BinaryWholeImage_Savoula.at<uchar>(iiLimit, jjLimit) == 255) && (DynamicWindowBinImage.at<uchar>(iiLimit, jjLimit) == 255) )
                            sumNeighBorThreshCnt++;
                    }
                }
            }
        }
        return sumNeighBorThreshCnt;
    }


    void BulkBinarization(string locAllImages) {

        DirectoryHandler *instance = DirectoryHandler::getInstance();

        // Get the image files
        vector<string> validImgFileNames;
        vector<string> validImgExtensions;
        validImgExtensions.push_back("jpg");
        if (boost::filesystem::exists(locAllImages))
            instance->getFilesInDirectory(locAllImages, validImgFileNames, validImgExtensions);

        for (auto const &imgFilePath: validImgFileNames) {
            boost::filesystem::path getImgPath(imgFilePath); // getting only the file name
            string imgFullPath = locAllImages + getImgPath.stem().string() + ".jpg";
            Mat imgOrig = imread(imgFullPath, CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_UNCHANGED);

            if (imgOrig.empty()) {
                std::cerr << "File not available for reading" << std::endl;
            }
            Mat imgBin = imgOrig.clone(); // keeping seperately the original image

            auto start = chrono::steady_clock::now();
            if (imgBin.channels() >= 3)
                cv::cvtColor(imgBin, imgBin, CV_BGR2GRAY);

            Mat correctedBinImag = imgBin;

            for (int iRow = 0; iRow < imgBin.rows; iRow++) {
                for (int jCol = 0; jCol < imgBin.cols; jCol++) {
                    if (imgBin.at<uchar>(iRow, jCol) > 180) {
                        correctedBinImag.at<uchar>(iRow, jCol) = 255;
                    } else {
                        correctedBinImag.at<uchar>(iRow, jCol) = 0;
                    }
                }
            }
            BasicAlgo::getInstance()->writeImageGivenPath(correctedBinImag, imgFullPath);
        }
    }


};

#endif //CLION_PROJECT_TEXTGRAPHICHSEPERATIONBULK_H

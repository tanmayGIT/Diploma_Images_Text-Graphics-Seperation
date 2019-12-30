

//
//  TextGraphicsSeperation.cpp
//  DocScanImageProcessing
//
//  Created by tmondal on 16/07/2018.
//  Copyright © 2018 Tanmoy. All rights reserved.
//


#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>

#include "Binarization/hdr/PopularDocBinarization.hpp"

#include "Binarization/hdr/Feature_Space_Partition_Binarization.h"
#include "Binarization/hdr/GatosBinarization.h"

#include "TextGraphichSeperationBulk.h"

#include <opencv2/ximgproc.hpp>
#include "opencv2/core/utility.hpp"
#include <highgui.h>
#include <cv.h>


// Copy the code from "19Jan_2019_TextGraphicsSeperation.txt" and replace from line 52 - 355, also uncomment the ln 481


using namespace cv::ximgproc;

// Size kernalSize(5, 5);
// RNG rng(12345);

void Init();



int main(int argc, char **argv) {

    Init();
}





void Init() {
    cv::Ptr<StructuredEdgeDetection> pDollar = createStructuredEdgeDetection("../model.yml");

    /*
     * This portion is for Bulk Images operations
     */
    TextGraphichSeperationBulk operateOnALL = *new TextGraphichSeperationBulk();

/*    string Image_Path_1 = "/home/mondal/Documents/Project_Work/DIBCO_2019_All_Isabelle/DIBCO_2019_Work/All-DIBCO-19-Papyri/All_Results_1/";
    operateOnALL.BulkBinarization(Image_Path_1);*/




/*   // For the binarization of DIBCO dataset
    string Image_Path_1 = "/home/mondal/Documents/Project_Work/Diploma_Images_Text-Graphics-Seperation/DIBCO/Dataset/AllImages/";
    string ResultSavingPath_1 = "/home/mondal/Documents/Project_Work/Diploma_Images_Text-Graphics-Seperation/DIBCO/Dataset/All_BackGd/";
    operateOnALL.ApplyBulkTextGraphicsSeperation(Image_Path_1, ResultSavingPath_1);*/









/*    string Image_Path_1 = "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/Orig_Images/";
    string ResultSavingPath_1 = "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/BG_Seperated/BG_Seperated_Result/";
    operateOnALL.ApplyBulkTextGraphicsSeperation(Image_Path_1, ResultSavingPath_1);*/

/*    string Image_Path_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/Orig_Images/";
    string ResultSavingPath_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/BG_Seperated/BG_Seperated_Result/";
    operateOnALL.ApplyBulkTextGraphicsSeperation(Image_Path_2, ResultSavingPath_2);*/

/*    string Image_Path_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/Orig_Images/";
    string ResultSavingPath_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BG_Seperated/BG_Seperated_Result/";
    operateOnALL.ApplyBulkTextGraphicsSeperation(Image_Path_3, ResultSavingPath_3);*/

/*    string Image_Path_4 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/Orig_Images/";
    string ResultSavingPath_4 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/BG_Seperated/BG_Seperated_Result/";
    operateOnALL.ApplyBulkTextGraphicsSeperation(Image_Path_4, ResultSavingPath_4);*/









/*    string Image_Path_0 = "/home/mondal/Documents/Project_Work/Diploma_Images_Text-Graphics-Seperation/DIBCO/Dataset/AllImages/";
    string Image_Path_0_Clus = "/home/mondal/Documents/Project_Work/Diploma_Images_Text-Graphics-Seperation/DIBCO/Dataset/All_Cluster_Results/";
    string ResultSavingPath_0 = "/home/mondal/Documents/Project_Work/Diploma_Images_Text-Graphics-Seperation/DIBCO/Dataset/All_Results/";
    string cannyImageSavingPath_0 = "/home/mondal/Documents/Project_Work/Diploma_Images_Text-Graphics-Seperation/DIBCO/Dataset/Canny_Results/";
    operateOnALL.ApplyFinalTextGraphicsRemoval(Image_Path_0, Image_Path_0_Clus, ResultSavingPath_0, cannyImageSavingPath_0);*/




/*    string Image_Path_1 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/Orig_Images/";
    string Image_Path_1_Clus =  "../../Dataset/Fujitsu_Seperated/300_DPI/Color/BG_Seperated/ClusteredImg/";
    string ResultSavingPath_1 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/BG_Seperated/Results/";
    string cannyImageSavingPath_1 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/BG_Seperated/Canny_Images/";
    operateOnALL.ApplyFinalTextGraphicsRemoval(Image_Path_1, Image_Path_1_Clus, ResultSavingPath_1, cannyImageSavingPath_1);*/


/*    string Image_Path_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/Orig_Images/";
    string Image_Path_2_Clus = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/BG_Seperated/ClusteredImg/";
    string ResultSavingPath_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/BG_Seperated/Results/";
    string cannyImageSavingPath_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/BG_Seperated/Canny_Images/";
    operateOnALL.ApplyFinalTextGraphicsRemoval(Image_Path_2, Image_Path_2_Clus, ResultSavingPath_2, cannyImageSavingPath_2 );*/


/*    string Image_Path_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/Orig_Images/";
    string Image_Path_3_Clus = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BG_Seperated/ClusteredImg/";
    string ResultSavingPath_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BG_Seperated/Results/";
    string cannyImageSavingPath_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BG_Seperated/Canny_Images/";

     operateOnALL.ApplyFinalTextGraphicsRemoval(Image_Path_3,Image_Path_3_Clus, ResultSavingPath_3, cannyImageSavingPath_3);*/

    // operateOnALL.ApplyFinalTextGraphicsRemoval_1(Image_Path_3,Image_Path_3_Clus, ResultSavingPath_3, cannyImageSavingPath );
    // operateOnALL.ApplyFinalTextGraphicsRemoval_2(Image_Path_3,Image_Path_3_Clus, ResultSavingPath_3, cannyImageSavingPath );



/*    string Image_Path_4 = "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/Orig_Images/";
    string Image_Path_4_Clus = "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/BG_Seperated/ClusteredImg/";
    string ResultSavingPath_4 =  "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/BG_Seperated/Results/";
    string cannyImageSavingPath_4 = "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/BG_Seperated/Canny_Images/";
    operateOnALL.ApplyFinalTextGraphicsRemoval(Image_Path_4, Image_Path_4_Clus, ResultSavingPath_4, cannyImageSavingPath_4);*/








    // To test the diploma image with savoula, niblack and wolf-jolin technique
    string Image_Path_1 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/Orig_Images/";
    string Image_Path_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/Orig_Images/";
    string Image_Path_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/Orig_Images/";
    string Image_Path_4 = "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/Orig_Images/";

    string ResultSavingPath_1 = "../../Dataset/Fujitsu_Seperated/300_DPI/Color/BinarizationResults/";
    string ResultSavingPath_2 = "../../Dataset/Fujitsu_Seperated/600_DPI/Color/BinarizationResults/";
    string ResultSavingPath_3 = "../../Dataset/Fujitsu_Seperated/300_DPI/Gray/BinarizationResults/";
    string ResultSavingPath_4 =  "../../Dataset/Fujitsu_Seperated/600_DPI/Gray/BinarizationResults/";

    operateOnALL.ApplyClassicBinarizationTechniques(Image_Path_1, ResultSavingPath_1);
    operateOnALL.ApplyClassicBinarizationTechniques(Image_Path_2, ResultSavingPath_2);
    operateOnALL.ApplyClassicBinarizationTechniques(Image_Path_3, ResultSavingPath_3);
    operateOnALL.ApplyClassicBinarizationTechniques(Image_Path_4, ResultSavingPath_4);







/*
    string Image_Path_1 = "/Volumes/Study_Materials/Dataset/Binarization/All_BackGd/";
    string ResultSavingPath_1 = "/Volumes/Study_Materials/Dataset/Binarization/All_Results_Grad/";
    operateOnALL.ApplyBulkEdgeDetection(Image_Path_1, ResultSavingPath_1);*/

/*    string Image_Path_1 = "/Volumes/Study_Materials/Dataset/Univ_Diploma/BDD3_JPEG/Scan_Fujitsu_Hidden_Seperated/300_DPI/Color/Ground_Truth/";
    operateOnALL.BulkBinarization(Image_Path_1);
    string Image_Path_2 = "/Volumes/Study_Materials/Dataset/Univ_Diploma/BDD3_JPEG/Scan_Fujitsu_Hidden_Seperated/300_DPI/Gray/Ground_Truth/";
    operateOnALL.BulkBinarization(Image_Path_2);
    string Image_Path_3 = "/Volumes/Study_Materials/Dataset/Univ_Diploma/BDD3_JPEG/Scan_Fujitsu_Hidden_Seperated/600_DPI/Gray/Ground_Truth/";
    operateOnALL.BulkBinarization(Image_Path_3);
    string Image_Path_4 = "/Volumes/Study_Materials/Dataset/Univ_Diploma/BDD3_JPEG/Scan_Fujitsu_Hidden_Seperated/600_DPI/Color/Ground_Truth/";
    operateOnALL.BulkBinarization(Image_Path_4);*/

}



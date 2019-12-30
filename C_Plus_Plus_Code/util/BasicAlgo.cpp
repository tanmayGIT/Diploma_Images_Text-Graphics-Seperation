/*
 * BasicAlgo.cpp
 *
 *  Created on: Feb 18, 2015
 *      Author: tanmoymondal
 */

#include "hdr/BasicAlgo.h"


BasicAlgo *BasicAlgo::instance = 0;

BasicAlgo *BasicAlgo::getInstance() {
    if (!instance)
        instance = new BasicAlgo();

    return instance;
}

BasicAlgo::BasicAlgo() {
}

BasicAlgo::~BasicAlgo() {
}

void BasicAlgo::showImage(cv::Mat &image) {
    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display Image", image);
    cvWaitKey(0);
    cvDestroyWindow("Display Image");
}

void BasicAlgo::writeImage(cv::Mat &image) {
    imwrite("/home/mondal/Documents/Save_Image.jpg", image);
}


void BasicAlgo::writeImageGivenPath(cv::Mat &image, string path) {
    try {
        imwrite(path, image);
    } catch (const std::exception &e) {
        std::cout << e.what(); // information from length_error printed
    }
}

/**
  *   @brief  Performs quick sort of a 1D array
  *
  *   @param  The pointer of 1D array
  *   @param  Start index of the array
  *   @param  End index of the array
  *   @return pointer of sorted 1D array
  */
int* BasicAlgo::quicksort (int *array, int start, int end )
{
    //	static unsigned int calls = 0;

    //cout << "QuickSort Call #: " << ++calls << endl;

    //function allows one past the end to be consistent with most function calls
    // but we normalize to left and right bounds that point to the data

    int leftbound = start;
    int rightbound = end - 1;

    if (rightbound <= leftbound )
        return NULL;

    int pivotIndex = leftbound + (rand() % (end - leftbound));
    int pivot = array[pivotIndex];

    // cout << " Pivot: " << "[" << pivotIndex << "] " << pivot << endl;
    int leftposition = leftbound;
    int rightposition = rightbound; // accounting for pivot that was moved out

    while ( leftposition < rightposition )
    {
        while ( leftposition < rightposition && array[leftposition] < pivot )
            ++leftposition;

        while ( rightposition > leftposition && array[rightposition] > pivot )
            --rightposition;

        if(leftposition < rightposition)
        {
            if (array[leftposition] != array[rightposition])
            {
                swap(array[leftposition],array[rightposition]);
                //		cout << " Swapping RightPosition: " << right position << " and LeftPosition: " << left position << endl;
            }
            else
                ++leftposition;
        }
    }

    // sort leaving the pivot out
    quicksort (array,leftbound, leftposition);  // left position is at the pivot which is one past the data
    quicksort (array,leftposition + 1,end);     // left position + 1 is past the pivot till the end
    return array;
}


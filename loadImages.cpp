#include "loadImages.hpp"

#include<bits/stdc++.h>
#include<Eigen/Dense>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

FaceData loadImages(const string& pathDir){
    vector<VectorXf> faceVec;
    vector<int> label;
    long long totalPixel = 0;

    for(auto& entry : filesystem::directory_iterator(pathDir)) {
        string filename = entry.path().filename().string();

        if(filename.rfind("subject", 0) == 0){
            // get subject number
            int subjectNum = stoi(filename.substr(7, filename.find('_') - 7));
            label.push_back(subjectNum);

            Mat image = imread(entry.path().string(), IMREAD_GRAYSCALE);

            // flatten image as a column vector, also normalize grayscale intensity
            if(!image.empty()){
                if(totalPixel == 0) totalPixel = image.rows * image.cols;

                VectorXf imgVec(totalPixel);
                for(int i = 0; i < image.rows; i++){
                    for(int j = 0; j < image.cols; j++)
                        imgVec(i * image.cols + j) = static_cast<float>(image.at<uchar>(i,j)) / 255.0f;
                }
                faceVec.push_back(imgVec);
            }
        }
    }
    FaceData dataMatrix;

    dataMatrix.matrix.resize(totalPixel, faceVec.size());
    for(size_t i = 0; i < faceVec.size(); i++){
        dataMatrix.matrix.col(i) = faceVec[i];
    }
    dataMatrix.label = label;

    return dataMatrix;
}
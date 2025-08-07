#pragma once

#include<bits/stdc++.h>
#include<Eigen/Dense>


struct FaceData{
    Eigen::MatrixXf matrix;
    std::vector<int> label;
};

FaceData loadImages(const std::string& pathDir);
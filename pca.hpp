#pragma once

#include<Eigen/Dense>

struct resultPCA{
    Eigen::MatrixXf eigenfaces;
    Eigen::VectorXf meanFace;
};

resultPCA performPCA(const Eigen::MatrixXf& data, int numComponents);
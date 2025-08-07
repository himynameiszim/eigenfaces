#include "loadImages.hpp"
#include "pca.hpp"

#include<bits/stdc++.h>
#include<Eigen/Dense>
#include<filesystem>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

int bestMatch(const VectorXf& testIMG, const resultPCA& pca, const MatrixXf& projection, const vector<int>& label){
    // center test img
    VectorXf centerTest = testIMG - pca.meanFace;
    // get weight in face space
    VectorXf testWeight = pca.eigenfaces.transpose() * centerTest;
    // get closest match
    float minDist = -1.0f;
    int bestMatchID = -1;

    for (int i = 0; i < projection.cols(); ++i) {
        // Euclidean difference between test and each training img
        float distance = (testWeight - projection.col(i)).norm();
        
        if (bestMatchID == -1 || distance < minDist) {
            minDist = distance;
            bestMatchID = i;
        }
    }
    
    return label[bestMatchID];
}

int main(){
    // input face dataset here, somehow opencv cannot read .gif images so i converted to png
    string pathDir = "/home/jimmy/Videos/eigen/eigenfaces/yale_data_png";
    FaceData data = loadImages(pathDir);
    
    if(data.matrix.size() == 0){
        cerr << "Load images failed from '" << pathDir << "'" << endl;
        return EXIT_FAILURE;
    }

    cout << "Loaded: " << data.matrix.cols() << " images." << endl;
    cout << "Matrix size: " << data.matrix.rows() << "x" << data.matrix.cols() << endl;

    int numComponents = 10;
    resultPCA pca = performPCA(data.matrix, numComponents);

    cout << "--- PCA results ---" << endl;
    cout << "Mean face size: " << pca.meanFace.size() << endl;
    cout << "Eigenfaces matrix size: " << pca.eigenfaces.rows() << "x" << pca.eigenfaces.cols() << endl;

    MatrixXf centerData = data.matrix.colwise() - pca.meanFace;
    MatrixXf projection = pca.eigenfaces.transpose() * centerData;

    cout << "--- Projection results ---" << endl;
    cout << "Projection matrix size: " << projection.rows() << "x" << projection.cols() << endl;

    int testImgID = 30;
    VectorXf testIMG = data.matrix.col(testImgID);
    int truthImgID = data.label[testImgID];

    // match
    int predictImgID = bestMatch(testIMG, pca, projection, data.label);
    cout << "--- Prediction results ---" << endl;
    cout << "Test ID: " << testImgID << "; Truth ID: " << truthImgID << "; Predicted ID: " << predictImgID << endl;
    cout << ((predictImgID == truthImgID) ? "Matched correctly!" : "Matched incorrectly!") << endl;

    return EXIT_SUCCESS;
}
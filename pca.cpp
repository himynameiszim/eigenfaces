#include "pca.hpp"

using namespace Eigen;

resultPCA performPCA(const MatrixXf& data, int numComponents){
    resultPCA result;

    // mean face is the average of all cols
    result.meanFace = data.rowwise().mean();

    // subtracting mean from each col to center data
    MatrixXf centerData = data.colwise() - result.meanFace;

    // perform SVD here, we only need left singular values (eigenfaces)
    JacobiSVD<MatrixXf> svd(centerData, Eigen::ComputeThinU);

    // only store top 'numComponents' eigenfaces    
    result.eigenfaces = svd.matrixU().leftCols(numComponents);

    return result;
}
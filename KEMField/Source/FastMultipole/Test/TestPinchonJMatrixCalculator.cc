#include "KFMLinearAlgebraDefinitions.hh"
#include "KFMMatrixOperations.hh"
#include "KFMPinchonGMatrixCalculator.hh"
#include "KFMPinchonJMatrixCalculator.hh"

#include <cmath>
#include <iostream>
#include <vector>

using namespace KEMField;

int main(int argc, char** argv)
{

    (void) argc;
    (void) argv;

    auto* calc = new KFMPinchonJMatrixCalculator();

    unsigned int deg = 3;
    calc->SetDegree(deg);

    std::vector<kfm_matrix*> matrices;
    matrices.clear();

    calc->AllocateMatrices(&matrices);
    calc->ComputeMatrices(&matrices);

    unsigned int l;
    unsigned int nrow;
    unsigned int ncol;

    for (unsigned int l = 0; l <= deg; l++) {
        nrow = 2 * l + 1;
        ncol = 2 * l + 1;

        std::cout << "------------------" << std::endl;
        std::cout << std::endl;
        for (unsigned int i = 0; i < nrow; i++) {
            for (unsigned int j = 0; j < ncol; j++) {
                std::cout << kfm_matrix_get(matrices.at(l), i, j) << "   ";
            }
            std::cout << std::endl;
        }
    }

    l = deg;
    nrow = 2 * l + 1;
    ncol = 2 * l + 1;

    //now we want to figure out want the SVD of J matrix looks like
    kfm_matrix* R = kfm_matrix_calloc(nrow, ncol);

    //now we are going to set the rows and columns according to the real and imag parts of R
    for (unsigned int i = 0; i < nrow; i++) {
        for (unsigned int j = 0; j < ncol; j++) {
            kfm_matrix_set(R, i, j, kfm_matrix_get(matrices.at(l), i, j));
        }
    }

    kfm_matrix* U = kfm_matrix_calloc(nrow, nrow);
    kfm_matrix* V = kfm_matrix_calloc(nrow, nrow);
    kfm_vector* s = kfm_vector_calloc(nrow);

    kfm_matrix_svd(R, U, s, V);

    for (unsigned int i = 0; i < nrow; i++) {
        std::cout << "singular value @ " << i << " = " << kfm_vector_get(s, i) << std::endl;
    }

    return 0;
}

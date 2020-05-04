#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMMessaging.hh"
#include "KFMVectorOperations.hh"

#include <cmath>
#include <iomanip>
#include <iostream>


using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{

    const unsigned int NVectors = 100;
    const unsigned int NVectorSize = 3;

    //allocate a vector
    kfm_vector* v1 = kfm_vector_alloc(NVectorSize);
    kfm_vector* v2 = kfm_vector_alloc(NVectorSize);
    kfm_vector* v3 = kfm_vector_alloc(NVectorSize);

    //allocate 4 matrices, to construct an euler rotation
    kfm_matrix* m1 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* m2 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* m3 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* m4 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* m4_inv = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* temp = kfm_matrix_calloc(NVectorSize, NVectorSize);

    //generate three angles
    double alpha = M_PI / 2.2313;
    double beta = M_PI / 4.934857;
    double gamma = M_PI / 3.9487;

    //construct some rotation matrices operating on Z, Y', Z''
    kfm_matrix_set_identity(m1);
    kfm_matrix_set(m1, 0, 0, std::cos(alpha));
    kfm_matrix_set(m1, 0, 1, -1.0 * std::sin(alpha));
    kfm_matrix_set(m1, 1, 0, std::sin(alpha));
    kfm_matrix_set(m1, 1, 1, std::cos(alpha));

    kfm_matrix_print(m1);

    kfm_matrix_set_identity(m2);
    kfm_matrix_set(m2, 0, 0, std::cos(beta));
    kfm_matrix_set(m2, 0, 2, -1.0 * std::sin(beta));
    kfm_matrix_set(m2, 2, 0, std::sin(beta));
    kfm_matrix_set(m2, 2, 2, std::cos(beta));

    kfm_matrix_print(m2);

    kfm_matrix_set_identity(m3);
    kfm_matrix_set(m3, 0, 0, std::cos(gamma));
    kfm_matrix_set(m3, 0, 1, -1.0 * std::sin(gamma));
    kfm_matrix_set(m3, 1, 0, std::sin(gamma));
    kfm_matrix_set(m3, 1, 1, std::cos(gamma));

    kfm_matrix_print(m3);

    kfm_matrix_set_identity(m4);
    kfm_matrix_set_identity(m4_inv);

    //compute the euler matrix
    kfm_matrix_multiply(m1, m4, temp);
    kfm_matrix_set(temp, m4);
    kfm_matrix_multiply(m2, m4, temp);
    kfm_matrix_set(temp, m4);
    kfm_matrix_multiply(m3, m4, temp);
    kfm_matrix_set(temp, m4);

    kfm_matrix_print(m4);

    //compute the euler matrix inverse
    kfm_matrix_multiply_with_transpose(true, false, m3, m4_inv, temp);
    kfm_matrix_set(temp, m4_inv);
    kfm_matrix_multiply_with_transpose(true, false, m2, m4_inv, temp);
    kfm_matrix_set(temp, m4_inv);
    kfm_matrix_multiply_with_transpose(true, false, m1, m4_inv, temp);
    kfm_matrix_set(temp, m4_inv);

    kfm_matrix_print(m4_inv);

    for (unsigned int n = 0; n < NVectors; n++) {
        //generate three points to make the triangle and compute centroid
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfm_vector_set(v1, j, ((double) rand() / (double) RAND_MAX));
        }
        kfm_vector_normalize(v1);

        //test the norm routine
        kfmout << "|v1| = " << kfm_vector_norm(v1) << kfmendl;

        kfmout << "v1 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v1, j) << ", ";
        }
        kfmout << kfmendl;

        //now test against series of discrete rotations
        kfm_vector_set(v1, v2);
        kfm_matrix_vector_product(m1, v2, v3);
        kfm_matrix_vector_product(m2, v3, v2);
        kfm_matrix_vector_product(m3, v2, v3);
        kfmout << "M3*M2*M1*v = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v3, j) << ", ";
        }
        kfmout << kfmendl;

        kfm_matrix_transpose_vector_product(m3, v3, v2);
        kfm_matrix_transpose_vector_product(m2, v2, v3);
        kfm_matrix_transpose_vector_product(m1, v3, v2);

        //multiply by series of inverse rotations
        kfmout << "(M1^T)*(M2^T)*(M3^T)*v = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v2, j) << ", ";
        }
        kfmout << kfmendl;

        kfmout << "----------------------------------------------------" << kfmendl;

        //multiply by full rotation
        kfm_vector_set(v1, v2);
        kfm_matrix_vector_product(m4, v2, v3);
        kfmout << "m4*v1 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v3, j) << ", ";
        }
        kfmout << kfmendl;

        //multiply by full inverse rotation
        kfm_matrix_vector_product(m4_inv, v3, v2);

        kfmout << "(m4^T)*(M4*v) = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v2, j) << ", ";
        }
        kfmout << kfmendl;

        kfmout << "----------------------------------------------------" << kfmendl;


        kfmout << "####################################################" << kfmendl;
    }


    //multiply rotation by inverse
    kfm_matrix_multiply(m4_inv, m4, temp);
    kfm_matrix_print(temp);


    return 0;
}

#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"
#include "KFMMessaging.hh"
#include "KFMVectorOperations.hh"

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>


using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{

    const unsigned int NVectors = 100;
    const unsigned int NVectorSize = 8;

    //temporary storage space
    double p1[NVectorSize];
    double p2[NVectorSize];
    double sum1;
    double sum2;

    //allocate a vector
    kfm_vector* v1 = kfm_vector_alloc(NVectorSize);
    kfm_vector* v2 = kfm_vector_alloc(NVectorSize);

    for (unsigned int n = 0; n < NVectors; n++) {
        sum1 = 0.0;
        sum2 = 0.0;

        //generate three points to make the triangle and compute centroid
        for (unsigned int j = 0; j < NVectorSize; j++) {
            p1[j] = ((double) rand() / (double) RAND_MAX);
            p2[j] = ((double) rand() / (double) RAND_MAX);

            sum1 += p1[j] * p1[j];
            sum2 += p2[j] * p2[j];
        }

        //normalize the test vectors
        sum1 = std::sqrt(sum1);
        sum2 = std::sqrt(sum2);
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfm_vector_set(v1, j, p1[j] / sum1);
            kfm_vector_set(v2, j, p2[j] / sum2);
        }

        //test the norm routine
        kfmout << "v1 norm = " << kfm_vector_norm(v1) << kfmendl;
        kfmout << "v2 norm = " << kfm_vector_norm(v2) << kfmendl;

        //test the add/substract routine
        kfmout << "v1 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v1, j) << ", ";
        }
        kfmout << kfmendl;
        kfm_vector_sub(v1, v2);
        kfmout << "v1 - v2 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v1, j) << ", ";
        }
        kfmout << kfmendl;
        kfm_vector_add(v1, v2);
        kfmout << "v1 - v2 + v2 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v1, j) << ", ";
        }
        kfmout << kfmendl;

        //test the scaling routine
        kfm_vector_scale(v1, 2.0);
        kfmout << "2.0*v1 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v1, j) << ", ";
        }
        kfmout << kfmendl;
        kfm_vector_scale(v1, 0.5);
        kfmout << "0.5*2.0*v1 = ";
        for (unsigned int j = 0; j < NVectorSize; j++) {
            kfmout << kfm_vector_get(v1, j) << ", ";
        }
        kfmout << kfmendl;


        //test inner product
        kfmout << " v1*v1 = " << kfm_vector_inner_product(v1, v1) << kfmendl;
        kfmout << " v1*v2 = " << kfm_vector_inner_product(v1, v2) << kfmendl;
        kfmout << " v2*v2 = " << kfm_vector_inner_product(v2, v2) << kfmendl;
    }


    return 0;
}

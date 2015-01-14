#include <iostream>
#include <cmath>
#include <iomanip>
#include <cstdlib>

#include "KFMMessaging.hh"
#include "KFMVectorOperations.hh"
#include "KFMMatrixOperations.hh"
#include "KFMMatrixVectorOperations.hh"


using namespace KEMField;

int main(int /*argc*/, char** /*argv*/)
{

    // const unsigned int NVectors = 100;
    const unsigned int NVectorSize = 3;

    //allocate a vector
    // kfm_vector* v1 = kfm_vector_alloc(NVectorSize);
    // kfm_vector* v2 = kfm_vector_alloc(NVectorSize);
    // kfm_vector* v3 = kfm_vector_alloc(NVectorSize);

    //allocate 4 matrices, to construct an euler rotation
    // kfm_matrix* m1 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    // kfm_matrix* m2 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    // kfm_matrix* m3 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* m4 = kfm_matrix_calloc(NVectorSize, NVectorSize);
    // kfm_matrix* m4_inv = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* temp = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* temp2 = kfm_matrix_calloc(NVectorSize, NVectorSize);

    //random matrix
    for(unsigned int i=0;i<3; i++)
    {
        for(unsigned int j=0;j<3; j++)
        {
            kfm_matrix_set(m4, i, j, ((double)rand()/(double)RAND_MAX) );
        }
    }



    kfmout<<"m4 = "<<kfmendl;
    kfm_matrix_print(m4);

    kfmout<<"--------------------------------------------------------"<<kfmendl;

    //now we are going to construct the SVD, and compute the psuedo inverse
    kfm_matrix* U = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* V = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* S = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_matrix* S_inv = kfm_matrix_calloc(NVectorSize, NVectorSize);
    kfm_vector* s = kfm_vector_calloc(NVectorSize);

    kfm_matrix_svd(m4, U, s, V);


    kfmout<<"U = "<<kfmendl;
    kfm_matrix_print(U);

    //construct the singular value matrix S and its inverse
    kfm_matrix_set_zero(S);
    kfm_matrix_set_zero(S_inv);
    double val;
    for(unsigned int i=0; i<NVectorSize; i++)
    {
        val = kfm_vector_get(s,i);
        kfm_matrix_set(S,i,i,val);

        if(val != 0.0)
        {
            //multiply 1/s against the i'th element of x
            val = (1.0/val);
            kfm_matrix_set(S_inv,i,i,val);
        }
        else
        {
            kfm_matrix_set(S_inv,i,i,0.0);
        }
    }

    kfmout<<"S = "<<kfmendl;
    kfm_matrix_print(S);

    kfmout<<"V = "<<kfmendl;
    kfm_matrix_print(V);

    //compute the psuedo inverse
    kfm_matrix_set_identity(temp);
    kfm_matrix_set_identity(temp2);
    kfm_matrix_multiply_with_transpose(true, false, U, temp, temp2);
    kfm_matrix_multiply(S_inv, temp2, temp);
    kfm_matrix_multiply(V, temp, temp2);

    kfmout<<"psuedo inverse of m4 = "<<kfmendl;
    kfm_matrix_print(temp2);

    //multiply
    kfmout<<"psuedo inverse(m4) * m4 = "<<kfmendl;
    kfm_matrix_multiply(temp2, m4, temp);
    kfm_matrix_print(temp);


    return 0;
}

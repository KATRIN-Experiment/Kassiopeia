#include "KFMPinchonJMatrixCalculator.hh"


namespace KEMField{


KFMPinchonJMatrixCalculator::KFMPinchonJMatrixCalculator()
{
    fDegree = 0;
    fGCalc = new KFMPinchonGMatrixCalculator();
}


KFMPinchonJMatrixCalculator::~KFMPinchonJMatrixCalculator()
{
    delete fGCalc;
}


void
KFMPinchonJMatrixCalculator::AllocateMatrices(std::vector< kfm_matrix* >* matrices)
{
    DeallocateMatrices(matrices);

    for(unsigned int l=0; l <= fDegree; l++)
    {
        unsigned int nrows = (2*l + 1);
        unsigned int ncolumns = (2*l + 1);
        kfm_matrix* m = NULL;
        m = kfm_matrix_alloc(nrows, ncolumns);
        matrices->push_back(m);
    }

}


void
KFMPinchonJMatrixCalculator::DeallocateMatrices(std::vector< kfm_matrix* >* matrices)
{
    for(unsigned int l=0; l < matrices->size(); l++)
    {
        if(matrices->at(l) != NULL)
        {
            kfm_matrix_free(matrices->at(l));
        }
    }
    matrices->clear();
}


bool
KFMPinchonJMatrixCalculator::ComputeMatrices(std::vector< kfm_matrix* >* matrices)
{
    if(matrices == NULL){return false;}

    kfm_matrix* j_target;
    kfm_matrix* j_prev;

    if( CheckMatrixSizes(matrices) )
    {
        //take care of the base cases first
        j_target = matrices->at(0);
        kfm_matrix_set_zero(j_target);
        kfm_matrix_set(j_target, 0, 0, 1.0);

        if(fDegree >= 1)
        {
            j_target = matrices->at(1);
            kfm_matrix_set_zero(j_target);
            kfm_matrix_set(j_target, 0, 1, -1.0);
            kfm_matrix_set(j_target, 1, 0, -1.0);
            kfm_matrix_set(j_target, 2, 2, 1.0);
        }

        for(unsigned int l=2; l <= fDegree; l++)
        {
            j_prev = matrices->at(l-1);
            j_target = matrices->at(l);
            ComputeJMatrixFromPrevious(l, j_prev, j_target);
        }

        return true;
    }
    else
    {
        return false;
    }
}

void
KFMPinchonJMatrixCalculator::ComputeJMatrixFromPrevious(unsigned int target_degree, kfm_matrix* prev, kfm_matrix* target)
{
    unsigned int l = target_degree - 1;
    double elem;

    kfm_matrix_set_zero(target);

    //first we need to allocate and compute the g matrices
    kfm_matrix* g_hat_z_inverse;
    unsigned int nrows = (2*l + 1);
    unsigned int ncolumns = (2*l + 1);
    g_hat_z_inverse = kfm_matrix_alloc(nrows,ncolumns);

    fGCalc->SetDegree(l);
    fGCalc->SetAsZHatInverse();
    fGCalc->ComputeMatrix(g_hat_z_inverse);

    kfm_matrix* g_y;
    nrows = (2*l + 3);
    ncolumns = (2*l + 1);
    g_y = kfm_matrix_alloc(nrows,ncolumns);

    fGCalc->SetDegree(l);
    fGCalc->SetAsY();
    fGCalc->ComputeMatrix(g_y);

    //now allocate a temporary matrix to store the result of prev*g_hat_z_inverse
    kfm_matrix* prod1;
    nrows = (2*l + 1);
    ncolumns = (2*l + 1);
    prod1 = kfm_matrix_alloc(nrows,ncolumns);

    //compute prev*g_hat_z_inverse
    kfm_matrix_multiply(prev, g_hat_z_inverse, prod1);
//    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, prev, g_hat_z_inverse, 0.0, prod1);

    //now allocate a temporary matrix to store the result of g_y*prod1
    kfm_matrix* prod2;
    nrows = (2*l + 3);
    ncolumns = (2*l + 1);
    prod2 = kfm_matrix_alloc(nrows,ncolumns);

    //compute g_y*prod1
    kfm_matrix_multiply(g_y, prod1, prod2);
//    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, g_y, prod1, 0.0, prod2);

    //now set the (2l+3) by (2l+1) submatrix of the target from prod2
    for(unsigned int i=0; i < 2*l + 3; i++)
    {
        for(unsigned int j=0; j < 2*l + 1; j++)
        {
            elem = kfm_matrix_get(prod2, i, j);
            kfm_matrix_set(target, i, j+1, elem);
        }
    }

    //now set the first and last column from the transpose of the first and last rows
    for(unsigned int i=0; i < 2*l + 3; i++)
    {
        //get element from first row
        elem = kfm_matrix_get(target, 0, i);
        //use it to set the first column
        kfm_matrix_set(target, i, 0, elem);
        //now do the same for last row/column
        elem = kfm_matrix_get(target, 2*l + 2, i);
        kfm_matrix_set(target, i, 2*l + 2, elem);
    }

    //now fill in the four corners
    kfm_matrix_set(target, 0, 0, 0);
    kfm_matrix_set(target, 0, 2*l + 2, 0);
    kfm_matrix_set(target, 2*l + 2, 0, 0);
    kfm_matrix_set(target, 2*l + 2, 2*l + 2, std::pow( (1.0/2.0) , (int)l) );

    //free the temporary matrices
    kfm_matrix_free(g_hat_z_inverse);
    kfm_matrix_free(g_y);
    kfm_matrix_free(prod1);
    kfm_matrix_free(prod2);
}


bool
KFMPinchonJMatrixCalculator::CheckMatrixSizes(std::vector< kfm_matrix* >* matrices)
{
    for(unsigned int l=0; l < matrices->size(); l++)
    {
        unsigned int nrows = (2*l + 1);
        unsigned int ncolumns = (2*l + 1);
        if(matrices->at(l) == NULL){return false;};
        if(nrows != matrices->at(l)->size1){return false;};
        if(ncolumns != matrices->at(l)->size2){return false;};
    }
    return true;
}

}//end of KEMField namespace

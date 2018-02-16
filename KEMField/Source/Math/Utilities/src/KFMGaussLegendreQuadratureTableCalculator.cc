#include "KFMGaussLegendreQuadratureTableCalculator.hh"

#include <cmath>

#include "KFMNumericalConstants.hh"
#include "KFMMatrixOperations.hh"
#include "KFMVectorOperations.hh"
#include "KFMMath.hh"

namespace KEMField
{

KFMGaussLegendreQuadratureTableCalculator::KFMGaussLegendreQuadratureTableCalculator()
{
    fJ = NULL; //symmetric matrix to be decomposed
    fLambda = NULL; //diagonal matrix of the eigenvalues
    fQ = NULL; //matrix of eigenvectors
    fQ_transpose = NULL; //transpose of fQ
    fWeights.clear();
    fAbscissa.clear();
    fN = 0;
}

KFMGaussLegendreQuadratureTableCalculator::~KFMGaussLegendreQuadratureTableCalculator()
{
        if(fJ != NULL){kfm_matrix_free(fJ); fJ = NULL;};
        if(fLambda != NULL){kfm_vector_free(fLambda); fLambda = NULL;};
        if(fQ != NULL){kfm_matrix_free(fQ); fQ = NULL;};
        if(fQ_transpose != NULL){kfm_matrix_free(fQ_transpose); fQ_transpose = NULL;};
}

void
KFMGaussLegendreQuadratureTableCalculator::SetNTerms(unsigned int n)
{
    if(n != 0 && fN != n)
    {
        fN = n;
        if(fJ != NULL){kfm_matrix_free(fJ); fJ = NULL;};
        if(fLambda != NULL){kfm_vector_free(fLambda); fLambda = NULL;};
        if(fQ != NULL){kfm_matrix_free(fQ); fQ = NULL;};
        if(fQ_transpose != NULL){kfm_matrix_free(fQ_transpose); fQ_transpose = NULL;};

        fJ = kfm_matrix_calloc(fN,fN);
        fLambda = kfm_vector_calloc(fN);
        fQ = kfm_matrix_calloc(fN,fN);
        fQ_transpose = kfm_matrix_calloc(fN,fN);

        fWeights.resize(fN);
        fAbscissa.resize(fN);
    }
}

void
KFMGaussLegendreQuadratureTableCalculator::Initialize()
{
    if(fN != 0)
    {
        //intialize the matrix fJ with the legendre coefficients
        double beta;

        for(unsigned int i=0; i<fN-1; i++)
        {
            beta = Beta(i+1);
            kfm_matrix_set(fJ, i, i+1, beta);
            kfm_matrix_set(fJ, i+1, i, beta);
        }

        //now since the matrix fJ is symmetric,
        //we can use the SVD routine to perform an eigenvalue/eigenvector to decompose it
        //however we have to be careful because we will not find negative eigenvalues
        //negative eigenvalues will appear as two instances of a positive eigenvalue
        kfm_matrix_svd(fJ, fQ, fLambda, fQ_transpose);

        double s;
        double w;

        for(unsigned int i=0; i<fN; i++)
        {
            //the singular value is the absolute value of the eigenvalue (abscissa)
            s = kfm_vector_get(fLambda,i);

            //the square of the first component of each eigenvector is the weighting factor
            w = kfm_matrix_get(fQ, 0, i);
            w = w*w;

            for(unsigned int j=0; j<i; j++)
            {
                if( std::fabs( fAbscissa[j] - s) < 100*KFM_EPSILON )
                {

                    //the eigenvalues are the abscissa, however the singular values
                    //are all positive, therefore if we see a repeated singular value X
                    //we replace it with -X,
                    s *= -1.0;

                    //if the abscissa corresponded to a negative eigenvalues
                    //then the weight will be zero, we fix this by replacing
                    //it with the weight of its positive partner

                    if( w > fWeights[j] )
                    {
                        fWeights[j] = w;
                    }
                    else
                    {
                        w = fWeights[j];
                    }

                    break;
                }
            }

            //check if s=0, if this is the case we have to compute the weight explicity
            //this only happens if we have an odd number of terms
            //See Chapter 25 of Abramowitz and Stegun, for calculation

            if(s == 0 )
            {
                double pn_derv =  -1.0*(fN+1)*KFMMath::ALP_nm((int)(fN+1), 0, 0);
                w = 2.0/(pn_derv*pn_derv);
            }

            fAbscissa[i] = s;
            fWeights[i] = w;

        }
    }
}

double
KFMGaussLegendreQuadratureTableCalculator::Beta(unsigned int i)
{
    double a = i;
    return std::sqrt( (a*a)/( (2.0*a - 1.0)*(2.0*a + 1.0) ) );
}

}

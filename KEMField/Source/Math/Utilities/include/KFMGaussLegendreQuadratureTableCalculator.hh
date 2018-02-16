#ifndef KFMGaussLegendreQuadratureTableCalculator_HH__
#define KFMGaussLegendreQuadratureTableCalculator_HH__

#include "KFMLinearAlgebraDefinitions.hh"
#include <vector>

namespace KEMField
{

/*
*
*@file KFMGaussLegendreQuadratureTableCalculator.hh
*@class KFMGaussLegendreQuadratureTableCalculator
*@brief Calculates the weights and abscissa of the n-th order Gauss-Legendre quadrature rule
*@details See the paper:
*
    @article
    {
        golub1969calculation,
        title={Calculation of Gauss quadrature rules},
        author={Golub, Gene H and Welsch, John H},
        journal={Mathematics of Computation},
        volume={23},
        number={106},
        pages={221--230},
        year={1969}
    }

*<b>Revision History:<b>
*Date Name Brief Description
*Mon Dec 16 22:26:21 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMGaussLegendreQuadratureTableCalculator
{
    public:
        KFMGaussLegendreQuadratureTableCalculator();
        virtual ~KFMGaussLegendreQuadratureTableCalculator();

        void SetNTerms(unsigned int n);

        void Initialize();

        void GetWeights(std::vector<double>* w) const {*w = fWeights;};
        void GetAbscissa(std::vector<double>* x) const {*x = fAbscissa;};

    private:

        double Beta(unsigned int i);

        unsigned int fN; //number terms in weight
        kfm_matrix* fJ; //symmetric matrix to be decomposed

        kfm_vector* fLambda; //vector of the eigenvalues
        kfm_matrix* fQ; //matrix of eigenvectors
        kfm_matrix* fQ_transpose; //transpose of fQ

        std::vector<double> fWeights;
        std::vector<double> fAbscissa;

};

}

#endif /* KFMGaussLegendreQuadratureTableCalculator_H__ */

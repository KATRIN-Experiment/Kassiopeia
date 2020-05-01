#ifndef KVMNumericalIntegrator_HH__
#define KVMNumericalIntegrator_HH__

#include "KFMArrayMath.hh"
#include "KVMField.hh"

#include <cstddef>
#include <vector>

namespace KEMField
{

/*
*
*@file KVMNumericalIntegrator.hh
*@class KVMNumericalIntegrator
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Dec 17 17:24:47 EST 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int domainDim, unsigned int rangeDim> class KVMNumericalIntegrator
{
  public:
    KVMNumericalIntegrator()
    {
        fWeights = nullptr;
        fAbscissa = nullptr;
    };

    virtual ~KVMNumericalIntegrator()
    {
        delete[] fWeights;
        delete[] fAbscissa;
    };

    //quadrature rules
    void SetNTerms(unsigned int n)
    {
        if (n != 0 && n != fN) {
            fN = n;
            if (fWeights != nullptr) {
                delete[] fWeights;
                fWeights = nullptr;
            };
            if (fAbscissa != nullptr) {
                delete[] fAbscissa;
                fAbscissa = nullptr;
            };

            fWeights = new double[fN];
            fAbscissa = new double[fN];

            fNEvaluations = 1;
            for (unsigned int i = 0; i < domainDim; i++) {
                fSize[i] = fN;
                fNEvaluations *= fN;
            }
        }
    }

    void SetWeights(const double* w)
    {
        for (unsigned int i = 0; i < fN; i++) {
            fWeights[i] = w[i];
        }
    }

    void SetAbscissa(const double* x)
    {
        for (unsigned int i = 0; i < fN; i++) {
            fAbscissa[i] = x[i];
        }
    }

    virtual void SetLowerLimits(const double* low)
    {
        for (unsigned int i = 0; i < domainDim; i++) {
            fLowerLimits[i] = low[i];
        }
    }

    virtual void SetUpperLimits(const double* up)
    {
        for (unsigned int i = 0; i < domainDim; i++) {
            fUpperLimits[i] = up[i];
        }
    }

    virtual void SetIntegrand(KVMField* integrand)
    {
        fIntegrand = integrand;
    };

    virtual void Integral(double* result) const
    {
        double prefactor = 1.0;

        for (unsigned int i = 0; i < domainDim; i++) {
            fHalfLimitDifference[i] = (fUpperLimits[i] - fLowerLimits[i]) / 2.0;
            fHalfLimitSum[i] = (fUpperLimits[i] + fLowerLimits[i]) / 2.0;
            prefactor *= fHalfLimitDifference[i];
        }

        for (unsigned int i = 0; i < rangeDim; i++) {
            fIntegrationResult[i] = 0.0;
        }

        double weight;
        for (unsigned int i = 0; i < fNEvaluations; i++) {
            //compute the spatial indices of the abscissa/weight point
            KFMArrayMath::RowMajorIndexFromOffset<domainDim>(i, fSize, fIndex);

            //compute the point where the evalution takes place and its corresponding weight
            weight = 1.0;
            for (unsigned int j = 0; j < domainDim; j++) {
                fEvaluationPoint[j] = fHalfLimitDifference[j] * fAbscissa[fIndex[j]] + fHalfLimitSum[j];
                weight *= fWeights[fIndex[j]];
            }

            //evaluate the integrand at the point
            fIntegrand->Evaluate(fEvaluationPoint, fFunctionResult);

            //for each of the variables in the range sum the weighted result
            for (unsigned int j = 0; j < rangeDim; j++) {
                fIntegrationResult[j] += weight * fFunctionResult[j];
            }
        }

        for (unsigned int j = 0; j < rangeDim; j++) {
            result[j] = prefactor * fIntegrationResult[j];
        }
    }

  protected:
    unsigned int fN;             //number of terms in 1-D quadrature
    unsigned int fNEvaluations;  //number of function evaluations
    unsigned int fSize[domainDim];

    double* fWeights;
    double* fAbscissa;

    double fLowerLimits[domainDim];
    double fUpperLimits[domainDim];

    KVMField* fIntegrand;

    //scratch space
    mutable double fHalfLimitDifference[domainDim];
    mutable double fHalfLimitSum[domainDim];

    mutable unsigned int fIndex[domainDim];

    mutable double fIntegrationResult[rangeDim];
    mutable double fEvaluationPoint[domainDim];
    mutable double fFunctionResult[rangeDim];
};


}  // namespace KEMField


#endif /* KVMNumericalIntegrator_H__ */

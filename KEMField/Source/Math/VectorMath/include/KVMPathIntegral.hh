#ifndef KVMPathIntegral_H
#define KVMPathIntegral_H

#include "KFMGaussLegendreQuadratureTableCalculator.hh"
#include "KVMCompactCurve.hh"
#include "KVMField.hh"
#include "KVMFieldWrapper.hh"
#include "KVMFixedArray.hh"
#include "KVMNumericalIntegrator.hh"

namespace KEMField
{


/**
*
*@file KVMPathIntegral.hh
*@class KVMPathIntegral
*@brief class to integrate a vector field with FieldNDIM components over a curve in R^3, each component is integrated separately
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul  6 11:53:35 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int FieldNDIM> class KVMPathIntegral
{
  public:
    KVMPathIntegral()
    {
        fNumInt = new KVMNumericalIntegrator<KVMCurveDDim, FieldNDIM>();
        fIntegrandWrapper =
            new KVMFieldWrapper<KVMPathIntegral, &KVMPathIntegral::Integrand>(this, KVMCurveDDim, FieldNDIM);
        fCurve = nullptr;
        fField = nullptr;
    }

    virtual ~KVMPathIntegral()
    {
        delete fNumInt;
        delete fIntegrandWrapper;
    }

    virtual void SetCurve(const KVMCompactCurve* aCurve)
    {
        fCurve = aCurve;
    };

    virtual void SetField(const KVMField* aField)
    {
        if (aField->GetNDimDomain() == KVMCurveRDim && aField->GetNDimRange() == FieldNDIM) {
            fField = aField;
        }
        else {
            fField = nullptr;
        }
    }

    virtual void SetNTerms(unsigned int n_quad)  //set number of terms in quadrature
    {
        KFMGaussLegendreQuadratureTableCalculator calc;
        calc.SetNTerms(n_quad);
        calc.Initialize();

        std::vector<double> w;
        std::vector<double> x;
        calc.GetWeights(&w);
        calc.GetAbscissa(&x);

        fNumInt->SetNTerms(n_quad);
        fNumInt->SetWeights(&(w[0]));
        fNumInt->SetAbscissa(&(x[0]));
    }

    virtual void Integral(double* result) const
    {
        //set the function to be integrated
        fNumInt->SetIntegrand(fIntegrandWrapper);

        //set the limits of integration
        fCurve->GetDomainBoundingBox(&fLow, &fHigh);
        fNumInt->SetLowerLimits(fLow.GetBareArray());
        fNumInt->SetUpperLimits(fHigh.GetBareArray());

        fNumInt->Integral(result);
    }

  protected:
    virtual void Integrand(const double* point, double* result) const
    {
        fVar[0] = point[0];

        InDomain = false;
        InDomain = fCurve->Evaluate(&fVar, &fP);  //get point
        InDomain = fCurve->Jacobian(&fVar, &fJ);  //get tangent at point

        if (InDomain) {
            fField->Evaluate(fP.GetBareArray(), result);

            double j_det = std::sqrt(fJ[0][0] * fJ[0][0] + fJ[0][1] * fJ[0][1] + fJ[0][2] * fJ[0][2]);
            for (unsigned int i = 0; i < FieldNDIM; i++) {
                result[i] *= j_det;
            }
        }
        else {
            for (unsigned int i = 0; i < FieldNDIM; i++) {
                result[i] = 0;
            }
        }
    }

    //the numerical integrator
    KVMNumericalIntegrator<KVMCurveDDim, FieldNDIM>* fNumInt;
    KVMFieldWrapper<KVMPathIntegral, &KVMPathIntegral::Integrand>* fIntegrandWrapper;

    //the curve we will integrate over
    const KVMCompactCurve* fCurve;

    //the field that is evaluated along the curve during integration
    const KVMField* fField;
    mutable double fFieldInput[KVMCurveRDim];
    mutable double fFieldOutput[FieldNDIM];

    //values used during integration that are variable
    //domain boundaries on the integration
    mutable double d;  //scratch space
    mutable double jacobian;
    mutable bool InDomain;
    mutable KVMFixedArray<double, KVMCurveDDim> fVar;
    mutable KVMFixedArray<double, KVMCurveRDim> fP;
    mutable KVMFixedArray<KVMFixedArray<double, KVMCurveRDim>, KVMCurveDDim> fJ;

    mutable KVMFixedArray<double, KVMCurveDDim> fLow;
    mutable KVMFixedArray<double, KVMCurveDDim> fHigh;
};


}  // namespace KEMField

#endif /* KVMPathIntegral_H */

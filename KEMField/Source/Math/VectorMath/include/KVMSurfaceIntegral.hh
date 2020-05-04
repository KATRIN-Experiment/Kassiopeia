#ifndef KVMSurfaceIntegral_H
#define KVMSurfaceIntegral_H

#include "KFMGaussLegendreQuadratureTableCalculator.hh"
#include "KVMCompactSurface.hh"
#include "KVMField.hh"
#include "KVMFieldWrapper.hh"
#include "KVMFixedArray.hh"
#include "KVMNumericalIntegrator.hh"

namespace KEMField
{

/**
*
*@file KVMSurfaceIntegral.hh
*@class KVMSurfaceIntegral
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Fri Jul  6 13:04:10 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int FieldNDIM> class KVMSurfaceIntegral
{
  public:
    KVMSurfaceIntegral()
    {
        //adaptive integrator is fastest is most low dimensional cases
        fNumInt = new KVMNumericalIntegrator<KVMSurfaceDDim, FieldNDIM>();
        fIntegrandWrapper =
            new KVMFieldWrapper<KVMSurfaceIntegral, &KVMSurfaceIntegral::Integrand>(this, KVMSurfaceDDim, FieldNDIM);
        fSurface = nullptr;
        fField = nullptr;
    }

    virtual ~KVMSurfaceIntegral()
    {
        delete fNumInt;
        delete fIntegrandWrapper;
    }

    virtual void SetSurface(const KVMCompactSurface* aSurface)
    {
        fSurface = aSurface;
    };

    virtual void SetField(const KVMField* aField)
    {
        if (aField->GetNDimDomain() == KVMSurfaceRDim && aField->GetNDimRange() == FieldNDIM) {
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
        fSurface->GetDomainBoundingBox(&fLow, &fHigh);
        fNumInt->SetLowerLimits(fLow.GetBareArray());
        fNumInt->SetUpperLimits(fHigh.GetBareArray());

        fNumInt->Integral(result);
    }

  protected:
    virtual void Integrand(const double* point, double* result) const
    {
        InDomain = false;
        fVar[0] = point[0];
        fVar[1] = point[1];
        InDomain = fSurface->Evaluate(&fVar, &fP);  //get point
        InDomain = fSurface->Jacobian(&fVar, &fJ);  //get jacobian

        if (InDomain) {
            fdU[0] = fJ[0][0];
            fdU[1] = fJ[0][1];
            fdU[2] = fJ[0][2];
            fdV[0] = fJ[1][0];
            fdV[1] = fJ[1][1];
            fdV[2] = fJ[1][2];
            ja_det = 0;
            ja_det += (fdU[1] * fdV[2] - fdU[2] * fdV[1]) * (fdU[1] * fdV[2] - fdU[2] * fdV[1]);
            ja_det += (fdU[2] * fdV[0] - fdU[0] * fdV[2]) * (fdU[2] * fdV[0] - fdU[0] * fdV[2]);
            ja_det += (fdU[0] * fdV[1] - fdU[1] * fdV[0]) * (fdU[0] * fdV[1] - fdU[1] * fdV[0]);
            ja_det = std::sqrt(ja_det);

            fField->Evaluate(fP.GetBareArray(), result);
            for (unsigned int i = 0; i < FieldNDIM; i++) {
                result[i] *= ja_det;
            }
        }
        else {
            for (unsigned int i = 0; i < FieldNDIM; i++) {
                result[i] = 0;
            }
        }
    }


    //the numerical integrator
    KVMNumericalIntegrator<KVMSurfaceDDim, FieldNDIM>* fNumInt;
    KVMFieldWrapper<KVMSurfaceIntegral, &KVMSurfaceIntegral::Integrand>* fIntegrandWrapper;

    //the surface to integrate over
    const KVMCompactSurface* fSurface;

    //the field defined on surface to be integrated
    const KVMField* fField;

    //values used during integration that are variable
    mutable double d;  //scratch space
    mutable double ja_det;
    mutable bool InDomain;
    mutable KVMFixedArray<double, KVMSurfaceDDim> fVar;
    mutable KVMFixedArray<double, KVMSurfaceRDim> fP;
    mutable KVMFixedArray<KVMFixedArray<double, KVMSurfaceRDim>, KVMSurfaceDDim> fJ;
    mutable KVMFixedArray<double, KVMSurfaceRDim> fdU;
    mutable KVMFixedArray<double, KVMSurfaceRDim> fdV;
    mutable KVMFixedArray<double, KVMSurfaceDDim> fLow;
    mutable KVMFixedArray<double, KVMSurfaceDDim> fHigh;
};


}  // namespace KEMField


#endif /* KVMSurfaceIntegral_H */

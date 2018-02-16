#include "KVMFluxIntegral.hh"

using namespace KEMField;

KVMFluxIntegral::KVMFluxIntegral():
KVMSurfaceIntegral<1>()
{
}

KVMFluxIntegral::~KVMFluxIntegral()
{
}

void
KVMFluxIntegral::SetField(const KVMField* aField)
{

    //field must take a point in R^3 to R^3
    if(aField->GetNDimDomain() == KVMSurfaceRDim && aField->GetNDimRange() == KVMSurfaceRDim)
    {
        fField = aField;
    }
    else
    {
        fField = NULL;
    }
}



void
KVMFluxIntegral::Integrand(const double* point, double* result) const
{
    InDomain = false;
    fVar[0] = point[0];
    fVar[1] = point[1];
    InDomain = fSurface->Evaluate(&fVar,&fP); //get point
    InDomain = fSurface->Jacobian(&fVar, &fJ); //get jacobian

    if(InDomain)
    {
        fField->Evaluate(fP.GetBareArray(),fV.GetBareArray());
        fdU[0] = fJ[0][0];
        fdU[1] = fJ[0][1];
        fdU[2] = fJ[0][2];
        fdV[0] = fJ[1][0];
        fdV[1] = fJ[1][1];
        fdV[2] = fJ[1][2];

        fN[0] = (fdU[1]*fdV[2] - fdU[2]*fdV[1]);
        fN[1] = (fdU[2]*fdV[0] - fdU[0]*fdV[2]);
        fN[2] = (fdU[0]*fdV[1] - fdU[1]*fdV[0]);

        result[0] = fN[0]*fV[0] + fN[1]*fV[1] + fN[2]*fV[2];
    }
    else
    {
        result[0] = 0;
    }

}

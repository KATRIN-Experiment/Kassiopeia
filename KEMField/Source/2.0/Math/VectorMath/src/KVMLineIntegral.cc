#include "KVMLineIntegral.hh"

using namespace KEMField;

KVMLineIntegral::KVMLineIntegral():
KVMPathIntegral<1>()
{

}


KVMLineIntegral::~KVMLineIntegral()
{

}

void
KVMLineIntegral::SetField(const KVMField* aField)
{
    //field must take a point in R^3 to R^3
    if(aField->GetNDimDomain() == KVMCurveRDim && aField->GetNDimRange() == KVMCurveRDim)
    {
        fField = aField;
    }
    else
    {
        fField = NULL;
    }
}

void
KVMLineIntegral::Integrand(const double* point, double* result) const
{
    fVar[0] = point[0];

    InDomain = false;
    InDomain = fCurve->Evaluate(&fVar,&fP); //get point
    InDomain = fCurve->Jacobian(&fVar,&fJ); //get tangent

    if(InDomain)
    {
        fField->Evaluate(fP.GetBareArray(), fV.GetBareArray());
        result[0] = fJ[0][0]*fV[0] + fJ[0][1]*fV[1] + fJ[0][1]*fV[2]; //form dot product between field and tangent
    }
    else
    {
        result[0] = 0;
    }
}

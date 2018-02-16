#include "KVMCompactCurve.hh"

#include <limits>

using namespace KEMField;

bool
KVMCompactCurve::PointInDomain(const KVMFixedArray<double, KVMCurveDDim >* in) const
{
    //default is to just check if we are in the domain bounding
    //box but the user can override this if they have a non-rectangular
    //domain, this is not-realy applicable to the domain of a curve
    //but we leave this ability in place.
    return InBoundingBox(in);
}

bool
KVMCompactCurve::Evaluate(const KVMFixedArray<double, KVMCurveDDim >* in,
KVMFixedArray<double, KVMCurveRDim >* out) const
{
    if(PointInDomain(in))
    {
        (*out)[0] = x( (*in)[0] );
        (*out)[1] = y( (*in)[0] );
        (*out)[2] = z( (*in)[0] );
        return true;
    }
    else
    {
        (*out)[0] = std::numeric_limits<double>::quiet_NaN();
        (*out)[1] = std::numeric_limits<double>::quiet_NaN();
        (*out)[2] = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
}

bool KVMCompactCurve::Jacobian(const KVMFixedArray<double, KVMCurveDDim >* in,
KVMFixedArray< KVMFixedArray<double, KVMCurveRDim>,  KVMCurveDDim>* jacobian) const
{
    if(PointInDomain(in))
    {
        (*jacobian)[0][0] = dxdu((*in)[0]);
        (*jacobian)[0][1] = dydu((*in)[0]);
        (*jacobian)[0][2] = dzdu((*in)[0]);
        return true;
    }
    else
    {
        (*jacobian)[0][0] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[0][1] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[0][2] = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
}

#include "KVMCompactVolume.hh"

#include <limits>

using namespace KEMField;


bool KVMCompactVolume::PointInDomain(const KVMFixedArray<double, KVMVolumeDDim>* in) const
{
    ///default is to just check if we are in the domain bounding
    ///box but the user can override this if they have a non-rectangular
    ///domain...however it should be noted that a rectangular domain is
    ///the best means of parameterizing an object since the numerical integrators
    ///have a hard time with functions that have sharp discontinuites (step function, etc.)
    return InBoundingBox(in);
}

bool KVMCompactVolume::Evaluate(const KVMFixedArray<double, KVMVolumeDDim>* in,
                                KVMFixedArray<double, KVMVolumeRDim>* out) const
{
    if (PointInDomain(in)) {
        (*out)[0] = x((*in)[0], (*in)[1], (*in)[2]);
        (*out)[1] = y((*in)[0], (*in)[1], (*in)[2]);
        (*out)[2] = z((*in)[0], (*in)[1], (*in)[2]);
        return true;
    }
    else {
        (*out)[0] = std::numeric_limits<double>::quiet_NaN();
        (*out)[1] = std::numeric_limits<double>::quiet_NaN();
        (*out)[2] = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
}


bool KVMCompactVolume::Jacobian(const KVMFixedArray<double, KVMVolumeDDim>* in,
                                KVMFixedArray<KVMFixedArray<double, KVMVolumeRDim>, KVMVolumeDDim>* jacobian) const
{
    if (PointInDomain(in)) {
        (*jacobian)[0][0] = dxdu((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[0][1] = dydu((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[0][2] = dzdu((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[1][0] = dxdv((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[1][1] = dydv((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[1][2] = dzdv((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[2][0] = dxdw((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[2][1] = dydw((*in)[0], (*in)[1], (*in)[2]);
        (*jacobian)[2][2] = dzdw((*in)[0], (*in)[1], (*in)[2]);
        return true;
    }
    else {
        (*jacobian)[0][0] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[0][1] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[0][2] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[1][0] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[1][1] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[1][2] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[2][0] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[2][1] = std::numeric_limits<double>::quiet_NaN();
        (*jacobian)[2][2] = std::numeric_limits<double>::quiet_NaN();
        return false;
    }
}

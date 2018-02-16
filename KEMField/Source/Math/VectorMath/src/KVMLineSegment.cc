#include "KVMLineSegment.hh"

using namespace KEMField;

KVMLineSegment::KVMLineSegment()
{

}

void KVMLineSegment::Initialize()
{
    fLowerLimits[0] = 0;
    fUpperLimits[0] = fL;
}

double KVMLineSegment::dxdu(const double& /*u*/) const {return fN[0];}
double KVMLineSegment::dydu(const double& /*u*/) const {return fN[1];}
double KVMLineSegment::dzdu(const double& /*u*/) const {return fN[2];}


double KVMLineSegment::x(const double& u) const {return fP1[0] + u*fN[0];}
double KVMLineSegment::y(const double& u) const {return fP1[1] + u*fN[1];}
double KVMLineSegment::z(const double& u) const {return fP1[2] + u*fN[2];}



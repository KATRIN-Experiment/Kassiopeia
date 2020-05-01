#include "KFMResponseKernel.hh"

namespace KEMField
{

KFMResponseKernel::KFMResponseKernel()
{
    ;
};

KFMResponseKernel::~KFMResponseKernel()
{
    ;
};

void KFMResponseKernel::SetSourceOrigin(double* origin)
{
    fSourceOrigin[0] = origin[0];
    fSourceOrigin[1] = origin[1];
    fSourceOrigin[2] = origin[2];

    fDel[0] = fSourceOrigin[0] - fTargetOrigin[0];
    fDel[1] = fSourceOrigin[1] - fTargetOrigin[1];
    fDel[2] = fSourceOrigin[2] - fTargetOrigin[2];
}

void KFMResponseKernel::SetTargetOrigin(double* origin)
{
    fTargetOrigin[0] = origin[0];
    fTargetOrigin[1] = origin[1];
    fTargetOrigin[2] = origin[2];

    fDel[0] = fSourceOrigin[0] - fTargetOrigin[0];
    fDel[1] = fSourceOrigin[1] - fTargetOrigin[1];
    fDel[2] = fSourceOrigin[2] - fTargetOrigin[2];
}


void KFMResponseKernel::SetDegree(int l_max)
{
    fDegree = std::fabs(l_max);
}

}  // namespace KEMField

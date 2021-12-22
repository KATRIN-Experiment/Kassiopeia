#include "KSTrajExactTrappedError.h"

using katrin::KThreeVector;

namespace Kassiopeia
{

KSTrajExactTrappedError::KSTrajExactTrappedError() :
    fTimeError(0.),
    fLengthError(0.),
    fPositionError(0., 0., 0.),
    fMomentumError(0., 0., 0.)
{}

KSTrajExactTrappedError::KSTrajExactTrappedError(const KSTrajExactTrappedError& anOperand) : KSMathArray<8>(anOperand)
{}

KSTrajExactTrappedError::~KSTrajExactTrappedError() = default;

const double& KSTrajExactTrappedError::GetTimeError() const
{
    fTimeError = fData[0];
    return fTimeError;
}
const double& KSTrajExactTrappedError::GetLengthError() const
{
    fLengthError = fData[1];
    return fLengthError;
}
const KThreeVector& KSTrajExactTrappedError::GetPositionError() const
{
    fPositionError.SetComponents(fData[2], fData[3], fData[4]);
    return fPositionError;
}
const KThreeVector& KSTrajExactTrappedError::GetMomentumError() const
{
    fMomentumError.SetComponents(fData[5], fData[6], fData[7]);
    return fMomentumError;
}

}  // namespace Kassiopeia

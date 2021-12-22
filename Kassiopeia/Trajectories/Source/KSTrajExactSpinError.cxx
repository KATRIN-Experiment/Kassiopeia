#include "KSTrajExactSpinError.h"

namespace Kassiopeia
{

KSTrajExactSpinError::KSTrajExactSpinError() :
    fTimeError(0.),
    fLengthError(0.),
    fPositionError(0., 0., 0.),
    fMomentumError(0., 0., 0.),
    fSpin0Error(0.),
    fSpinError(0., 0., 0.)
{}

KSTrajExactSpinError::KSTrajExactSpinError(const KSTrajExactSpinError& anOperand) : KSMathArray<12>(anOperand) {}

KSTrajExactSpinError::~KSTrajExactSpinError() = default;

const double& KSTrajExactSpinError::GetTimeError() const
{
    fTimeError = fData[0];
    return fTimeError;
}
const double& KSTrajExactSpinError::GetLengthError() const
{
    fLengthError = fData[1];
    return fLengthError;
}
const katrin::KThreeVector& KSTrajExactSpinError::GetPositionError() const
{
    fPositionError.SetComponents(fData[2], fData[3], fData[4]);
    return fPositionError;
}
const katrin::KThreeVector& KSTrajExactSpinError::GetMomentumError() const
{
    fMomentumError.SetComponents(fData[5], fData[6], fData[7]);
    return fMomentumError;
}
const double& KSTrajExactSpinError::GetSpin0Error() const
{
    fSpin0Error = fData[8];
    return fSpin0Error;
}
const katrin::KThreeVector& KSTrajExactSpinError::GetSpinError() const
{
    fSpinError.SetComponents(fData[9], fData[10], fData[11]);
    return fSpinError;
}

}  // namespace Kassiopeia

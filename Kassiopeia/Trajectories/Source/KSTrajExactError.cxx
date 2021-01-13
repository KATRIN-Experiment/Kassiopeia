#include "KSTrajExactError.h"

namespace Kassiopeia
{

KSTrajExactError::KSTrajExactError() :
    fTimeError(0.),
    fLengthError(0.),
    fPositionError(0., 0., 0.),
    fMomentumError(0., 0., 0.)
{}

KSTrajExactError::KSTrajExactError(const KSTrajExactError& anOperand) : KSMathArray<8>(anOperand) {}

KSTrajExactError::~KSTrajExactError() = default;

const double& KSTrajExactError::GetTimeError() const
{
    fTimeError = fData[0];
    return fTimeError;
}
const double& KSTrajExactError::GetLengthError() const
{
    fLengthError = fData[1];
    return fLengthError;
}
const KGeoBag::KThreeVector& KSTrajExactError::GetPositionError() const
{
    fPositionError.SetComponents(fData[2], fData[3], fData[4]);
    return fPositionError;
}
const KGeoBag::KThreeVector& KSTrajExactError::GetMomentumError() const
{
    fMomentumError.SetComponents(fData[5], fData[6], fData[7]);
    return fMomentumError;
}

}  // namespace Kassiopeia

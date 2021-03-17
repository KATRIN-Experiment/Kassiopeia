#include "KSTrajMagneticError.h"

#include <cmath>

namespace Kassiopeia
{

KSTrajMagneticError::KSTrajMagneticError() : fTimeError(0.), fLengthError(0.), fPositionError(0., 0, 0.) {}

KSTrajMagneticError::KSTrajMagneticError(const KSTrajMagneticError& anOperand) : KSMathArray<5>(anOperand) {}

KSTrajMagneticError::~KSTrajMagneticError() = default;

const double& KSTrajMagneticError::GetTimeError() const
{
    fTimeError = fData[0];
    return fTimeError;
}
const double& KSTrajMagneticError::GetLengthError() const
{
    fLengthError = fData[1];
    return fLengthError;
}
const KGeoBag::KThreeVector& KSTrajMagneticError::GetPositionError() const
{
    fPositionError.SetComponents(fData[2], fData[3], fData[4]);
    return fPositionError;
}

}  // namespace Kassiopeia

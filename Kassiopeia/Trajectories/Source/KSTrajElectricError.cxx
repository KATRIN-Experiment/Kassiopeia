#include "KSTrajElectricError.h"

#include <cmath>

namespace Kassiopeia
{

KSTrajElectricError::KSTrajElectricError() : fTimeError(0.), fLengthError(0.), fPositionError(0., 0, 0.) {}

KSTrajElectricError::KSTrajElectricError(const KSTrajElectricError& anOperand) : KSMathArray<5>(anOperand) {}

KSTrajElectricError::~KSTrajElectricError() = default;

const double& KSTrajElectricError::GetTimeError() const
{
    fTimeError = fData[0];
    return fTimeError;
}
const double& KSTrajElectricError::GetLengthError() const
{
    fLengthError = fData[1];
    return fLengthError;
}
const katrin::KThreeVector& KSTrajElectricError::GetPositionError() const
{
    fPositionError.SetComponents(fData[2], fData[3], fData[4]);
    return fPositionError;
}

}  // namespace Kassiopeia

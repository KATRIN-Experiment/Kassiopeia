#include "KSTrajMagneticError.h"

#include <cmath>

namespace Kassiopeia
{

    KSTrajMagneticError::KSTrajMagneticError() :
        fTimeError( 0. ),
        fLengthError( 0. ),
        fPositionError( 0., 0, 0. )
    {
    }

    KSTrajMagneticError::~KSTrajMagneticError()
    {
    }

    const double& KSTrajMagneticError::GetTimeError() const
    {
        fTimeError = fData[0];
        return fTimeError;
    }
    const double& KSTrajMagneticError::GetLengthError() const
    {
        fLengthError = fData[ 1 ];
        return fLengthError;
    }
    const KThreeVector& KSTrajMagneticError::GetPositionError() const
    {
        fPositionError.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fPositionError;
    }

}

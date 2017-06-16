#include "KSTrajAdiabaticSpinError.h"

namespace Kassiopeia
{

    KSTrajAdiabaticSpinError::KSTrajAdiabaticSpinError() :
        fTimeError( 0. ),
        fLengthError( 0. ),
        fPositionError( 0., 0., 0. ),
        fMomentumError( 0., 0., 0. )
    {
    }

    KSTrajAdiabaticSpinError::~KSTrajAdiabaticSpinError()
    {
    }

    const double& KSTrajAdiabaticSpinError::GetTimeError() const
    {
        fTimeError = fData[0];
        return fTimeError;
    }
    const double& KSTrajAdiabaticSpinError::GetLengthError() const
    {
        fLengthError = fData[ 1 ];
        return fLengthError;
    }
    const KThreeVector& KSTrajAdiabaticSpinError::GetPositionError() const
    {
        fPositionError.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fPositionError;
    }
    const KThreeVector& KSTrajAdiabaticSpinError::GetMomentumError() const
    {
        fMomentumError.SetComponents( fData[ 5 ], fData[ 6 ], fData[ 7 ] );
        return fMomentumError;
    }
    const double& KSTrajAdiabaticSpinError::GetAlignedSpinError() const
    {
        fAlignedSpinError = fData[8];
        return fAlignedSpinError;
    }
    const double& KSTrajAdiabaticSpinError::GetSpinAngleError() const
    {
        fSpinAngleError = fData[ 9 ];
        return fSpinAngleError;
    }

}

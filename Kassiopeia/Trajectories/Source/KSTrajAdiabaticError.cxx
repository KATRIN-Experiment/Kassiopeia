#include "KSTrajAdiabaticError.h"

#include <cmath>

namespace Kassiopeia
{

    KSTrajAdiabaticError::KSTrajAdiabaticError() :
            fTimeError( 0. ),
            fLengthError( 0. ),
            fGuidingCenterPositionError( 0., 0., 0. ),
            fLongitudinalMomentumError( 0. ),
            fTransverseMomentumError( 0. ),
            fPhaseError( 0. )
    {
    }
    KSTrajAdiabaticError::KSTrajAdiabaticError( const KSTrajAdiabaticError& anOperand ) :
        KSMathArray< 8 >( anOperand )
    {
    }
    KSTrajAdiabaticError::~KSTrajAdiabaticError()
    {
    }

    const double& KSTrajAdiabaticError::GetTimeError() const
    {
        fTimeError = fData[ 0 ];
        return fTimeError;
    }
    const double& KSTrajAdiabaticError::GetLengthError() const
    {
        fLengthError = fData[ 1 ];
        return fLengthError;
    }
    const KThreeVector& KSTrajAdiabaticError::GetGuidingCenterPositionError() const
    {
        fGuidingCenterPositionError.SetComponents( fData[ 2 ], fData[ 3 ], fData[ 4 ] );
        return fGuidingCenterPositionError;
    }
    const double& KSTrajAdiabaticError::GetLongitudinalMomentumError() const
    {
        fLongitudinalMomentumError = fData[ 5 ];
        return fLongitudinalMomentumError;
    }
    const double& KSTrajAdiabaticError::GetTransverseMomentumError() const
    {
        fTransverseMomentumError = fData[ 6 ];
        return fTransverseMomentumError;
    }
    const double& KSTrajAdiabaticError::GetPhaseError() const
    {
        fPhaseError = fData[ 7 ];
        return fPhaseError;
    }

}

#include "KTwoVector.hh"

namespace KGeoBag
{

    const KTwoVector KTwoVector::sInvalid( NAN, NAN );
    const KTwoVector KTwoVector::sZero( 0., 0. );

    const KTwoVector KTwoVector::sXUnit( 1., 0. );
    const KTwoVector KTwoVector::sYUnit( 0., 1. );

    const KTwoVector KTwoVector::sZUnit( 1., 0. );
    const KTwoVector KTwoVector::sRUnit( 0., 1. );

    KTwoVector::KTwoVector()
    {
        fData[0] = 0.;
        fData[1] = 0.;
    }

}

#ifndef KGTRANSFORMATIONBUILDER_HH_
#define KGTRANSFORMATIONBUILDER_HH_

#include "KComplexElement.hh"

#include "KTransformation.hh"
#include "KThreeVector.hh"

using namespace KGeoBag;
namespace katrin
{
    typedef KComplexElement< KTransformation > KGTransformationBuilder;

    template< >
    inline bool KGTransformationBuilder::AddAttribute( KContainer* aContainer )
    {
        if( (aContainer->GetName() == "displacement") || (aContainer->GetName() == "d") )
        {
            KThreeVector* tVector = NULL;
            aContainer->ReleaseTo( tVector );
            fObject->SetDisplacement( tVector->X(), tVector->Y(), tVector->Z() );
            delete tVector;
            return true;
        }
        if( (aContainer->GetName() == "rotation_euler") || (aContainer->GetName() == "r_eu") )
        {
            KThreeVector& tVector = aContainer->AsReference< KThreeVector >();
            fObject->SetRotationEuler( tVector.X(), tVector.Y(), tVector.Z() );
            return true;
        }
        if( (aContainer->GetName() == "rotation_axis_angle") || (aContainer->GetName() == "r_aa") )
        {
            KThreeVector& tVector = aContainer->AsReference< KThreeVector >();
            fObject->SetRotationAxisAngle( tVector.X(), tVector.Y(), tVector.Z() );
            return true;
        }

        return false;
    }
}

#endif

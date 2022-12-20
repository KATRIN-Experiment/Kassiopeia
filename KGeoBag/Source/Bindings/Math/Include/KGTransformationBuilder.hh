#ifndef KGTRANSFORMATIONBUILDER_HH_
#define KGTRANSFORMATIONBUILDER_HH_

#include "KComplexElement.hh"
#include "KThreeVector.hh"
#include "KTransformation.hh"

namespace katrin
{
typedef KComplexElement<KTransformation> KGTransformationBuilder;

template<> inline bool KGTransformationBuilder::AddAttribute(KContainer* aContainer)
{
    if ((aContainer->GetName() == "displacement") || (aContainer->GetName() == "d")) {
        KThreeVector* tVector = nullptr;
        aContainer->ReleaseTo(tVector);
        fObject->SetDisplacement(tVector->X(), tVector->Y(), tVector->Z());
        delete tVector;
        return true;
    }
    if ((aContainer->GetName() == "rotation_euler") || (aContainer->GetName() == "r_eu")) {
        auto& tVector = aContainer->AsReference<KThreeVector>();
        fObject->SetRotationEuler(tVector.X(), tVector.Y(), tVector.Z());
        return true;
    }
    if ((aContainer->GetName() == "rotation_axis_angle") || (aContainer->GetName() == "r_aa")) {
        auto& tVector = aContainer->AsReference<KThreeVector>();
        fObject->SetRotationAxisAngle(tVector.X(), tVector.Y(), tVector.Z());
        return true;
    }

    return false;
}
}  // namespace katrin

#endif

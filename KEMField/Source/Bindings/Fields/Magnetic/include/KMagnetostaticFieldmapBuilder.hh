/*
 * KMagnetostaticFieldmapBuilder.hh
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#ifndef KMAGNETOSTATICFIELDMAPBUILDER_HH_
#define KMAGNETOSTATICFIELDMAPBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KEMStreamableThreeVector.hh"
#include "KToolbox.h"

#ifdef KEMFIELD_USE_VTK
#include "KMagnetostaticFieldmap.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KMagnetostaticFieldmap> KMagnetostaticFieldmapBuilder;

template<> inline bool KMagnetostaticFieldmapBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "directory") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmap::SetDirectory);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmap::SetFile);
        return true;
    }
    if (aContainer->GetName() == "interpolation") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmap::SetInterpolation);
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////

typedef KComplexElement<KEMField::KMagnetostaticFieldmapCalculator> KMagnetostaticFieldmapCalculatorBuilder;

template<> inline bool KMagnetostaticFieldmapCalculatorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "directory") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetDirectory);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetFile);
        return true;
    }
    if (aContainer->GetName() == "force_update") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetForceUpdate);
        return true;
    }
    if (aContainer->GetName() == "compute_gradient") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetComputeGradient);
        return true;
    }
    if (aContainer->GetName() == "center") {
        KEMField::KEMStreamableThreeVector center;
        aContainer->CopyTo(center);
        fObject->SetCenter(center.GetThreeVector());
        return true;
    }
    if (aContainer->GetName() == "length") {
        KEMField::KEMStreamableThreeVector length;
        aContainer->CopyTo(length);
        fObject->SetLength(length.GetThreeVector());
        return true;
    }
    if (aContainer->GetName() == "mirror_x") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetMirrorX);
        return true;
    }
    if (aContainer->GetName() == "mirror_y") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetMirrorY);
        return true;
    }
    if (aContainer->GetName() == "mirror_z") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetMirrorZ);
        return true;
    }
    if (aContainer->GetName() == "spacing") {
        aContainer->CopyTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::SetSpacing);
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsReference<std::string>());
        std::vector<KGeoBag::KGSpace*>::const_iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            BINDINGMSG(eWarning) << "no spaces found for specifier <" << aContainer->AsReference<std::string>() << ">"
                                 << eom;
            return false;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddSpace(tSpace);
        }
        return true;
    }
    if (aContainer->GetName() == "field") {
        std::string fieldName;
        aContainer->CopyTo(fieldName);
        KEMField::KMagnetostaticField* field =
            katrin::KToolbox::GetInstance().Get<KEMField::KMagnetostaticField>(fieldName);
        fObject->AddMagneticField(field);
        return true;
    }
    return false;
}

template<> inline bool KMagnetostaticFieldmapCalculatorBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KEMField::KMagnetostaticField>()) {
        aContainer->ReleaseTo(fObject, &KEMField::KMagnetostaticFieldmapCalculator::AddMagneticField);
        return true;
    }
    return false;
}

template<> inline bool KMagnetostaticFieldmapCalculatorBuilder::End()
{
    fObject->Initialize();
    return true;
}

} /* namespace katrin */

#else /* KEMFIELD_USE_VTK */

// dummy for bindings when fieldmap is not available
#include "KMagnetostaticField.hh"

#include <limits>

namespace KEMField
{
class KMagnetostaticFieldmap : public KMagnetostaticField
{
    KThreeVector MagneticPotentialCore(const KPosition& /*P*/) const override
    {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return KThreeVector(nan, nan, nan);
    }
    KThreeVector MagneticFieldCore(const KPosition& /*P*/) const override
    {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return KThreeVector(nan, nan, nan);
    }
    KGradient MagneticGradientCore(const KPosition& /*P*/) const override
    {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return KGradient(nan, nan, nan, nan, nan, nan, nan, nan, nan);
    }
};
class KMagnetostaticFieldmapCalculator
{};
}  // namespace KEMField

namespace katrin
{

typedef KComplexElement<KEMField::KMagnetostaticFieldmap> KMagnetostaticFieldmapBuilder;

template<> inline bool KMagnetostaticFieldmapBuilder::AddAttribute(KContainer* /*aContainer*/)
{
    return true;
}

template<> inline bool KMagnetostaticFieldmapBuilder::End()
{
    BINDINGMSG(eWarning) << "KEMField is installed without VTK. Fieldmap is not supported!" << eom;
    return true;
}

typedef KComplexElement<KEMField::KMagnetostaticFieldmapCalculator> KMagnetostaticFieldmapCalculatorBuilder;

template<> inline bool KMagnetostaticFieldmapCalculatorBuilder::AddAttribute(KContainer* /*aContainer*/)
{
    return true;
}

template<> inline bool KMagnetostaticFieldmapCalculatorBuilder::End()
{
    BINDINGMSG(eWarning) << "KEMField is installed without VTK. Fieldmap is not supported!" << eom;
    return true;
}

} /* namespace katrin */

#endif /* KEMFIELD_USE_VTK */

#endif /* KMAGNETOSTATICFIELDMAPBUILDER_HH_ */

/*
 * KElectricPotentialmapBuilder.hh
 *
 *  Created on: 27 May 2016
 *      Author: wolfgang
 */

#ifndef KELECTROSTATICPOTENTIALMAPBUILDER_HH_
#define KELECTROSTATICPOTENTIALMAPBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KEMStreamableThreeVector.hh"
#include "KToolbox.h"

#ifdef KEMFIELD_USE_VTK
#include "KElectrostaticPotentialmap.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KElectrostaticPotentialmap> KElectrostaticPotentialmapBuilder;

template<> inline bool KElectrostaticPotentialmapBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "directory") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmap::SetDirectory);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmap::SetFile);
        return true;
    }
    if (aContainer->GetName() == "interpolation") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmap::SetInterpolation);
        return true;
    }
    return false;
}

////////////////////////////////////////////////////////////////////

typedef KComplexElement<KEMField::KElectrostaticPotentialmapCalculator> KElectrostaticPotentialmapCalculatorBuilder;

template<> inline bool KElectrostaticPotentialmapCalculatorBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        this->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "directory") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetDirectory);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetFile);
        return true;
    }
    if (aContainer->GetName() == "force_update") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetForceUpdate);
        return true;
    }
    if (aContainer->GetName() == "compute_field") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetComputeField);
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
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetMirrorX);
        return true;
    }
    if (aContainer->GetName() == "mirror_y") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetMirrorY);
        return true;
    }
    if (aContainer->GetName() == "mirror_z") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetMirrorZ);
        return true;
    }
    if (aContainer->GetName() == "spacing") {
        aContainer->CopyTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::SetSpacing);
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        std::vector<KGeoBag::KGSpace*>::const_iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            BINDINGMSG(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
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
        KEMField::KElectrostaticField* field =
            katrin::KToolbox::GetInstance().Get<KEMField::KElectrostaticField>(fieldName);
        fObject->AddElectricField(field);
        return true;
    }
    return false;
}

template<> inline bool KElectrostaticPotentialmapCalculatorBuilder::AddElement(KContainer* aContainer)
{
    if (aContainer->Is<KEMField::KElectrostaticField>()) {
        aContainer->ReleaseTo(fObject, &KEMField::KElectrostaticPotentialmapCalculator::AddElectricField);
        return true;
    }
    return false;
}

template<> inline bool KElectrostaticPotentialmapCalculatorBuilder::End()
{
    fObject->Initialize();
    return true;
}

} /* namespace katrin */

#else /* KEMFIELD_USE_VTK */

// dummy for bindings when potentialmap is not available
#include "KElectrostaticField.hh"

#include <limits>

namespace KEMField
{
class KElectrostaticPotentialmap : public KElectrostaticField
{
    double PotentialCore(const KPosition& /*P*/) const override
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    KFieldVector ElectricFieldCore(const KPosition& /*P*/) const override
    {
        double nan = std::numeric_limits<double>::quiet_NaN();
        return KFieldVector(nan, nan, nan);
    }
};
class KElectrostaticPotentialmapCalculator
{};
}  // namespace KEMField

namespace katrin
{

typedef KComplexElement<KEMField::KElectrostaticPotentialmap> KElectrostaticPotentialmapBuilder;

template<> inline bool KElectrostaticPotentialmapBuilder::AddAttribute(KContainer* /*aContainer*/)
{
    return true;
}

template<> inline bool KElectrostaticPotentialmapBuilder::End()
{
    BINDINGMSG(eWarning) << "KEMField is installed without VTK. Potentialmap is not supported!" << eom;
    return true;
}

using KElectrostaticPotentialmapCalculatorBuilder = KComplexElement<KEMField::KElectrostaticPotentialmapCalculator>;

template<> inline bool KElectrostaticPotentialmapCalculatorBuilder::AddAttribute(KContainer* /*aContainer*/)
{
    return true;
}

template<> inline bool KElectrostaticPotentialmapCalculatorBuilder::End()
{
    BINDINGMSG(eWarning) << "KEMField is installed without VTK. Potentialmap is not supported!" << eom;
    return true;
}

} /* namespace katrin */

#endif /* KEMFIELD_USE_VTK */

#endif /* KELECTROSTATICPOTENTIALMAPBUILDER_HH_ */

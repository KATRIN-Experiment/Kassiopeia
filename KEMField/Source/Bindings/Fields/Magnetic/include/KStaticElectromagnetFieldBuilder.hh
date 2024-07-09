/*
 * KStaticElectromagnetFieldBuilder.hh
 *
 *  Created on: 26 Mar 2016
 *      Author: wolfgang
 */

#ifndef KSTATICELECTROMAGNETFIELDBUILDER_HH_
#define KSTATICELECTROMAGNETFIELDBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KGStaticElectromagnetField.hh"
#include "KIntegratingMagnetostaticFieldSolver.hh"

namespace katrin
{

typedef KComplexElement<KEMField::KGStaticElectromagnetField> KStaticElectromagnetFieldBuilder;

template<> inline bool KStaticElectromagnetFieldBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string name;
        aContainer->CopyTo(name);
        SetName(name);
        fObject->SetName(name);
        return true;
    }
    if (aContainer->GetName() == "directory") {
        aContainer->CopyTo(fObject, &KEMField::KStaticElectromagnetField::SetDirectory);
        return true;
    }
    if (aContainer->GetName() == "file") {
        aContainer->CopyTo(fObject, &KEMField::KStaticElectromagnetField::SetFile);
        return true;
    }
    if (aContainer->GetName() == "save_magfield3") {
        aContainer->CopyTo(fObject, &KEMField::KGStaticElectromagnetField::SetSaveMagfield3);
        return true;
    }
    if (aContainer->GetName() == "directory_magfield3") {
        aContainer->CopyTo(fObject, &KEMField::KGStaticElectromagnetField::SetDirectoryMagfield3);
        return true;
    }
    if (aContainer->GetName() == "surfaces") {
        std::vector<KGeoBag::KGSurface*> tSurfaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSurfaces(aContainer->AsString());

        std::vector<KGeoBag::KGSurface*>::const_iterator tSurfaceIt;
        KGeoBag::KGSurface* tSurface;

        if (tSurfaces.size() == 0) {
            BINDINGMSG(eWarning) << "no surfaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSurfaceIt = tSurfaces.begin(); tSurfaceIt != tSurfaces.end(); tSurfaceIt++) {
            tSurface = *tSurfaceIt;
            fObject->AddSurface(tSurface);
        }
        return true;
    }
    if (aContainer->GetName() == "spaces") {
        std::vector<KGeoBag::KGSpace*> tSpaces =
            KGeoBag::KGInterface::GetInstance()->RetrieveSpaces(aContainer->AsString());
        std::vector<KGeoBag::KGSpace*>::const_iterator tSpaceIt;
        KGeoBag::KGSpace* tSpace;

        if (tSpaces.size() == 0) {
            BINDINGMSG(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return true;
        }

        for (tSpaceIt = tSpaces.begin(); tSpaceIt != tSpaces.end(); tSpaceIt++) {
            tSpace = *tSpaceIt;
            fObject->AddSpace(tSpace);
        }
        return true;
    }
    if (aContainer->GetName() == "system") {
        KGeoBag::KGSpace* tSpace = KGeoBag::KGInterface::GetInstance()->RetrieveSpace(aContainer->AsString());

        if (tSpace == nullptr) {
            BINDINGMSG(eWarning) << "no spaces found for specifier <" << aContainer->AsString() << ">" << eom;
            return false;
        }

        fObject->SetSystem(tSpace);

        return true;
    }
    return false;
}

template<> inline bool KStaticElectromagnetFieldBuilder::AddElement(KContainer* anElement)
{
    if (anElement->Is<KEMField::KMagneticFieldSolver>()) {
        if (!(fObject->GetFieldSolver())) {
            std::shared_ptr<KEMField::KMagneticFieldSolver> solver;
            anElement->ReleaseTo(solver);
            fObject->SetFieldSolver(solver);
        }
        else {
            BINDINGMSG(eError) << "Cannot set more than one magnetic field"
                                  " solver for field "
                               << GetName() << "!" << eom;
        }
    }
    else
        return false;
    return true;
}

template<> inline bool KStaticElectromagnetFieldBuilder::End()
{
    if (!(fObject->GetFieldSolver())) {
        BINDINGMSG(eWarning) << " No magnetic field solver set in field "
                           << GetName() << " - falling back to integrating solver!" << eom;
        auto solver = std::make_shared<KEMField::KIntegratingMagnetostaticFieldSolver>();
        fObject->SetFieldSolver(solver);
        return true;
    }
    else
        return true;
}

} /* namespace katrin */

#endif /* KSTATICELECTROMAGNETFIELDBUILDER_HH_ */

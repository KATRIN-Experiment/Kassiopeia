/*
 * FieldFinder.cxx
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#include "KSFieldFinder.h"

#include "KElectricField.hh"
#include "KMagneticField.hh"
#include "KSElectricKEMField.h"
#include "KSFieldsMessage.h"
#include "KSMagneticKEMField.h"
#include "KToolbox.h"

using namespace katrin;

namespace Kassiopeia
{

KSElectricField* getElectricField(std::string name)
{
    fieldmsg_debug("Trying to retrieve electric field <" << name << "> from toolbox" << eom);

    if (KToolbox::GetInstance().Get<KSElectricField>(name) != nullptr) {
        return KToolbox::GetInstance().Get<KSElectricField>(name);
    }

    if (KToolbox::GetInstance().Get<KSElectricField>(name + "_") != nullptr)  // alias
    {
        return KToolbox::GetInstance().Get<KSElectricField>(name + "_");
    }

    if (KToolbox::GetInstance().Get<KEMField::KElectricField>(name) != nullptr) {
        auto newObject = new KSElectricKEMField(KToolbox::GetInstance().Get<KEMField::KElectricField>(name));
        newObject->SetName(name + "_");
        KToolbox::GetInstance().GetInstance().Add(newObject);
        return newObject;
    }

    fieldmsg(eError) << "Electric field <" << name << "> does not exist in toolbox" << eom;
    return nullptr;
}

KSMagneticField* getMagneticField(std::string name)
{
    fieldmsg_debug("Trying to retrieve magnetic field <" << name << "> from toolbox" << eom);

    if (KToolbox::GetInstance().Get<KSMagneticField>(name) != nullptr) {
        return KToolbox::GetInstance().Get<KSMagneticField>(name);
    }

    if (KToolbox::GetInstance().Get<KSMagneticField>(name + "_") != nullptr)  // alias
    {
        return KToolbox::GetInstance().Get<KSMagneticField>(name + "_");
    }

    if (KToolbox::GetInstance().Get<KEMField::KMagneticField>(name) != nullptr) {
        auto newObject = new KSMagneticKEMField(KToolbox::GetInstance().Get<KEMField::KMagneticField>(name));
        newObject->SetName(name + "_");
        KToolbox::GetInstance().GetInstance().Add(newObject);
        return newObject;
    }

    fieldmsg(eError) << "Magnetic field <" << name << "> does not exist in toolbox" << eom;
    return nullptr;
}

std::vector<KSElectricField*> getAllElectricFields()
{
    fieldmsg_debug("Trying to retrieve all electric fields from toolbox" << eom);

    auto fields = KToolbox::GetInstance().GetAll<KSElectricField>();
    if (fields.empty()) {
        for (auto entry : KToolbox::GetInstance().GetAll<KEMField::KElectricField>()) {
            //This is a glaring memory leak
            fields.push_back(new KSElectricKEMField(entry));
        }
    }
    return fields;
}

std::vector<KSMagneticField*> getAllMagneticFields()
{
    fieldmsg_debug("Trying to retrieve all magnetic fields from toolbox" << eom);

    auto fields = KToolbox::GetInstance().GetAll<KSMagneticField>();
    if (fields.empty()) {
        for (auto entry : KToolbox::GetInstance().GetAll<KEMField::KMagneticField>()) {
            //This is a glaring memory leak
            fields.push_back(new KSMagneticKEMField(entry));
        }
    }
    return fields;
}

}  // namespace Kassiopeia

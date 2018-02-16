/*
 * FieldFinder.cxx
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#include "KSFieldFinder.h"
#include "KSFieldsMessage.h"
#include "KToolbox.h"
#include "KEMToolbox.hh"

#include "KElectricField.hh"
#include "KMagneticField.hh"

#include "KSElectricKEMField.h"
#include "KSMagneticKEMField.h"

using namespace katrin;
using namespace KEMField;

namespace Kassiopeia {

KSElectricField* getElectricField(std::string name)
{
    fieldmsg_debug("Trying to retrieve electric field <" << name << "> from toolbox" << eom);

    if (KToolbox::GetInstance().Get<KSElectricField>(name) != nullptr)
    {
        return KToolbox::GetInstance().Get<KSElectricField>(name);
    }

    if (KToolbox::GetInstance().Get<KElectricField>(name) != nullptr)
    {
        //This is a glaring memory leak
        return new KSElectricKEMField(KToolbox::GetInstance().Get<KElectricField>(name));
    }

    fieldmsg( eError ) << "Electric field <" << name << "> does not exist in toolbox" << eom;
    return nullptr;
}

KSMagneticField* getMagneticField(std::string name)
{
    fieldmsg_debug("Trying to retrieve magnetic field <" << name << "> from toolbox" << eom);

    if (KToolbox::GetInstance().Get<KSMagneticField>(name) != nullptr)
    {
        return KToolbox::GetInstance().Get<KSMagneticField>(name);
    }

    if (KToolbox::GetInstance().Get<KMagneticField>(name) != nullptr)
    {
        //This is a glaring memory leak
        return new KSMagneticKEMField(KToolbox::GetInstance().Get<KMagneticField>(name));
    }

    fieldmsg( eError ) << "Magnetic field <" << name << "> does not exist in toolbox" << eom;
    return nullptr;
}

std::vector< KSMagneticField* > getAllMagneticFields()
{
    fieldmsg_debug("Trying to retrieve all magnetic fields from toolbox" << eom);

    auto fields = KToolbox::GetInstance().GetAll<KSMagneticField>();
    if (fields.empty())
    {
        for (auto entry : KToolbox::GetInstance().GetAll<KMagneticField>())
        {
            //This is a glaring memory leak
            fields.push_back(new KSMagneticKEMField(entry));
        }
    }
    return fields;
}

} //Kassiopeia



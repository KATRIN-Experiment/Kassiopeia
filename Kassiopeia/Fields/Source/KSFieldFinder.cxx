/*
 * FieldFinder.cxx
 *
 *  Created on: 6 Aug 2015
 *      Author: wolfgang
 */

#include "KSFieldFinder.h"
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
    if(KToolbox::GetInstance().Get<KSElectricField>(name))
        return KToolbox::GetInstance().Get<KSElectricField>(name);

    //This is a glaring memory leak
    return new KSElectricKEMField(KToolbox::GetInstance().Get<KElectricField>(name));
}

KSMagneticField* getMagneticField(std::string name)
{
    if(KToolbox::GetInstance().Get<KSMagneticField>(name))
        return KToolbox::GetInstance().Get<KSMagneticField>(name);

    //This is a glaring memory leak
    return new KSMagneticKEMField(KToolbox::GetInstance().Get<KMagneticField>(name));
}

std::vector< KSMagneticField* > getAllMagneticFields()
{
    auto fields = KToolbox::GetInstance().GetAll<KSMagneticField>();
    if(fields.empty()) {
        for (auto entry : KToolbox::GetInstance().GetAll<KMagneticField>()) {
            //This is a glaring memory leak
            fields.push_back(new KSMagneticKEMField(entry));
        }
    }
    return fields;
}

} //Kassiopeia



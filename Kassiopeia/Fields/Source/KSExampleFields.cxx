/*
 * KSExampleFields.cpp
 *
 *  Created on: 23 Aug 2016
 *      Author: wolfgang
 */

#include "KSExampleFields.h"

#include "KElectrostaticConstantField.hh"
#include "KMagnetostaticConstantField.hh"
#include "KSElectricKEMField.h"
#include "KSMagneticKEMField.h"

using namespace KEMField;

namespace Kassiopeia
{


KSElectricField* MakeConstantElectricField(KGeoBag::KThreeVector field)
{
    auto* kemfield = new KElectrostaticConstantField();
    kemfield->SetField(field);

    auto* kasfield = new KSElectricKEMField();
    kasfield->SetElectricField(kemfield);
    return kasfield;
}

KSMagneticField* MakeConstantMagneticField(KGeoBag::KThreeVector field)
{
    auto* kemfield = new KMagnetostaticConstantField();
    kemfield->SetField(field);

    auto* kasfield = new KSMagneticKEMField();
    kasfield->SetMagneticField(kemfield);
    return kasfield;
}

} /* namespace Kassiopeia */

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

namespace Kassiopeia {


KSElectricField* MakeConstantElectricField(
        KGeoBag::KThreeVector field)
{
    KElectrostaticConstantField* kemfield = new KElectrostaticConstantField();
    kemfield->SetField(field);

    KSElectricKEMField* kasfield = new KSElectricKEMField();
    kasfield->SetElectricField(kemfield);
    return kasfield;
}

KSMagneticField* MakeConstantMagneticField(
        KGeoBag::KThreeVector field)
{
    KMagnetostaticConstantField* kemfield = new KMagnetostaticConstantField();
    kemfield->SetField(field);

    KSMagneticKEMField* kasfield = new KSMagneticKEMField();
    kasfield->SetMagneticField(kemfield);
    return kasfield;
}

} /* namespace Kassiopeia */

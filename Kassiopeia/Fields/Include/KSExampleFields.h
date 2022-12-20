/*
 * KSExampleFields.h
 *
 *  Created on: 23 Aug 2016
 *      Author: wolfgang
 */

#ifndef KASSIOPEIA_FIELDS_INCLUDE_KSEXAMPLEFIELDS_H_
#define KASSIOPEIA_FIELDS_INCLUDE_KSEXAMPLEFIELDS_H_

#include "KSElectricField.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

KSElectricField* MakeConstantElectricField(katrin::KThreeVector field);

KSMagneticField* MakeConstantMagneticField(katrin::KThreeVector field);

} /* namespace Kassiopeia */

#endif /* KASSIOPEIA_FIELDS_INCLUDE_KSEXAMPLEFIELDS_H_ */

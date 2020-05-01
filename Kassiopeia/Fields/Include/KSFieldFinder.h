#ifndef KSFIELD_FINDER_H_
#define KSFIELD_FINDER_H_
/*
 * KSFieldFinder.h
 *
 *  Created on: 19.05.2015
 *      Author: gosda
 */

#include "KSElectricField.h"
#include "KSMagneticField.h"

namespace Kassiopeia
{

KSElectricField* getElectricField(std::string name);
KSMagneticField* getMagneticField(std::string name);

std::vector<KSElectricField*> getAllElectricFields();
std::vector<KSMagneticField*> getAllMagneticFields();

}  // namespace Kassiopeia

#endif /* KSFIELD_FINDER_H_ */

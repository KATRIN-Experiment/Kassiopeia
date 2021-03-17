/*
 * KMagnetostaticConstantFieldBuilder.cc
 *
 *  Created on: 23 Mar 2016
 *      Author: wolfgang
 */

#include "KMagnetostaticConstantFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KSFieldMagneticConstantBuilder::~KComplexElement() = default;

STATICINT sKSFieldMagneticConstant =
    KEMToolboxBuilder::ComplexElement<KMagnetostaticConstantField>("constant_magnetic_field");

STATICINT sKSFieldMagneticConstantStructure =
    KSFieldMagneticConstantBuilder::Attribute<std::string>("name") +
    KSFieldMagneticConstantBuilder::Attribute<KEMStreamableThreeVector>("field") +
    KSFieldMagneticConstantBuilder::Attribute<KEMStreamableThreeVector>("location");


} /* namespace katrin */

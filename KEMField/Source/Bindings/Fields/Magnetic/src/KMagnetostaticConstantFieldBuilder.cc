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
    //! the name of the magnetic field.
    KSFieldMagneticConstantBuilder::Attribute<std::string>("name") +
    //! the cartesian components of the magnetic field vector in tesla.
    KSFieldMagneticConstantBuilder::Attribute<KEMStreamableThreeVector>("field") +
    KSFieldMagneticConstantBuilder::Attribute<KEMStreamableThreeVector>("location");


} /* namespace katrin */

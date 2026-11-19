/*
 * KMagneticDipoleFieldBuilder.cc
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#include "KMagneticDipoleFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KMagneticDipoleFieldBuilder::~KComplexElement() = default;

STATICINT sKMagneticDipoleFieldStructure =
    //! the name of the magnetic field.
    KMagneticDipoleFieldBuilder::Attribute<std::string>("name") +
    //! cartesian components of the location of the dipole.
    KMagneticDipoleFieldBuilder::Attribute<KEMStreamableThreeVector>("location") +
    //! the components of the dipole moment vector in ampere * meter^2.
    KMagneticDipoleFieldBuilder::Attribute<KEMStreamableThreeVector>("moment");

STATICINT sKMagneticDipoleField = KEMToolboxBuilder::ComplexElement<KMagneticDipoleField>("magnetic_dipole_field");

} /* namespace katrin */

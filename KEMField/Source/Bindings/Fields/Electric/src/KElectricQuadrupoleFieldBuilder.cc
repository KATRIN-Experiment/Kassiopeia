/*
 * KElectricQuadrupoleFieldBuilder.cc
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#include "KElectricQuadrupoleFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KElectricQuadrupoleFieldBuilder::~KComplexElement() = default;

STATICINT sKElectricQuadrupoleFieldStructure =
    //! the name of the electric field.
    KElectricQuadrupoleFieldBuilder::Attribute<std::string>("name") +
    //! the vertex of the asymptotic cones.
    KElectricQuadrupoleFieldBuilder::Attribute<KEMStreamableThreeVector>("location") +
    //! V0 in the equation of the element description, in volts.
    KElectricQuadrupoleFieldBuilder::Attribute<double>("strength") +
    //! z0 in the equation of the element description, in meters.
    KElectricQuadrupoleFieldBuilder::Attribute<double>("length") +
    //! r0 in the equation of the element description, in meters.
    KElectricQuadrupoleFieldBuilder::Attribute<double>("radius");

STATICINT sKToolboxStructure = KEMToolboxBuilder::ComplexElement<KElectricQuadrupoleField>("electric_quadrupole_field");


} /* namespace katrin */

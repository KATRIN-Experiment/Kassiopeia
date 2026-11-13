/*
 * KStaticElectromagnetFieldBuilder.cc
 *
 *  Created on: 26 Mar 2016
 *      Author: wolfgang
 */

#include "KStaticElectromagnetFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"

using namespace KEMField;
using namespace std;

namespace katrin
{

template<> KStaticElectromagnetFieldBuilder::~KComplexElement() = default;

STATICINT sKStaticElectromagnetFieldStructure =
    //! the name of the magnetic field.
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("name") +
    //! name of file in which to cache computation.
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("file") +
    //! name of directory in which to cache computation.
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("directory") +
    KStaticElectromagnetFieldBuilder::Attribute<bool>("save_magfield3") +
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("directory_magfield3") +
    //! path to exactly one kgeobag space whose coordinate system the
    //! computation will use.
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("system") +
    //! paths to kgeobag surfaces which are equipped with both a
    //! discretization and electromagnetic properties.
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("surfaces") +
    //! paths to kgeobag surfaces which are equipped with both a
    //! discretization of the boundary and electromagnetic properties.
    KStaticElectromagnetFieldBuilder::Attribute<std::string>("spaces");

STATICINT sKStaticElectromagnetField =
    KEMToolboxBuilder::ComplexElement<KGStaticElectromagnetField>("electromagnet_field");
} /* namespace katrin */

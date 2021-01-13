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

STATICINT sKStaticElectromagnetFieldStructure = KStaticElectromagnetFieldBuilder::Attribute<std::string>("name") +
                                                KStaticElectromagnetFieldBuilder::Attribute<std::string>("file") +
                                                KStaticElectromagnetFieldBuilder::Attribute<std::string>("directory") +
                                                KStaticElectromagnetFieldBuilder::Attribute<bool>("save_magfield3") +
                                                KStaticElectromagnetFieldBuilder::Attribute<std::string>("system") +
                                                KStaticElectromagnetFieldBuilder::Attribute<std::string>("surfaces") +
                                                KStaticElectromagnetFieldBuilder::Attribute<std::string>("spaces");

STATICINT sKStaticElectromagnetField =
    KEMToolboxBuilder::ComplexElement<KGStaticElectromagnetField>("electromagnet_field");
} /* namespace katrin */

/*
 * KElectrostaticBoundaryFieldBuilder.cc
 *
 *  Created on: 17 Jun 2015
 *      Author: wolfgang
 */
#include "KElectrostaticBoundaryFieldBuilder.hh"

#include "KEMToolboxBuilder.hh"


using namespace KEMField;
namespace katrin
{

template<> KElectrostaticBoundaryFieldBuilder::~KComplexElement() = default;

STATICINT sKEMToolBoxBuilder = KEMToolboxBuilder::ComplexElement<KGElectrostaticBoundaryField>("electrostatic_field");

STATICINT sKElectrostaticBoundaryFieldBuilder =
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("name") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("directory") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("file") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("system") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("surfaces") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("spaces") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("symmetry") +
    KElectrostaticBoundaryFieldBuilder::Attribute<unsigned int>("hash_masked_bits") +
    KElectrostaticBoundaryFieldBuilder::Attribute<double>("hash_threshold") +
    KElectrostaticBoundaryFieldBuilder::Attribute<double>("minimum_element_area") +
    KElectrostaticBoundaryFieldBuilder::Attribute<double>("maximum_element_aspect_ratio");
}  // namespace katrin

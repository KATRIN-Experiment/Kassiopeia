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
    //! the name of the electric field.
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("name") +
    //! name of directory in which to cache computation. No directory should
    //! default to the KEMField cache.
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("directory") +
    //! name of file in which to cache computation.
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("file") +
    //! path to exactly one kgeobag space whose coordinate system the
    //! computation will use.
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("system") +
    //! paths to KGeoBag surfaces which are equipped with both a
    //! discretization and electromagnetic properties.
    //! Surfaces not fulfilling both criteria are ignored.
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("surfaces") +
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("spaces") +
    //! may be none, axial or discrete_axial, which constrains methods
    //! that may be used in computation. The symmetry is always defined along
    //! the z-axis.
    KElectrostaticBoundaryFieldBuilder::Attribute<std::string>("symmetry") +
    //! this value is passed to the charge density solver
    //! to configure the retrival mechanism for precomputed solutions
    //! of the charge density solver.
    //! You should not have to use this option.
    KElectrostaticBoundaryFieldBuilder::Attribute<unsigned int>("hash_masked_bits") +
    //! this value is passed to the charge density solver
    //! to configure the retrival mechanism for precomputed solutions
    //! of the charge density solver.
    //! You should not have to use this option.
    KElectrostaticBoundaryFieldBuilder::Attribute<double>("hash_threshold") +
    //! elements of the geometry with an area below this value are ignored
    //! in the electric field computation
    KElectrostaticBoundaryFieldBuilder::Attribute<double>("minimum_element_area") +
    //! elements with an aspect ratio above this value are ignored in the
    //! electric field computation.
    //! The aspect ratio of a triangle is the ratio of the longest side over
    //! the corresponding height.
    //! The aspect ratio of a rectangle is the ratio of the longer side over
    //! the shorter side.
    //! The aspect ratio of a wire is the length divided by the diameter.
    KElectrostaticBoundaryFieldBuilder::Attribute<double>("maximum_element_aspect_ratio");
}  // namespace katrin

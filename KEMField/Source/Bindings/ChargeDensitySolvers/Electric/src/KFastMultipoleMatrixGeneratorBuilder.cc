/*
 * KFastMultipoleMatrixGeneratorBuilder.cc
 *
 *  Created on: 18 Aug 2015
 *      Author: wolfgang
 */

#include "KFastMultipoleMatrixGeneratorBuilder.hh"

#include "KKrylovChargeDensitySolverBuilder.hh"
#include "KKrylovPreconditionerGeneratorBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KFastMultipoleMatrixGeneratorBuilder::~KComplexElement() = default;

STATICINT sKFastMultipoleMatrixGeneratorStructure =
    //! @values numeric, analytic, rwg, reference, default
    KFastMultipoleMatrixGeneratorBuilder::Attribute<std::string>("integrator") +
    //! strategy for the construction of the tree of cubes. The options are:
    //!
    //! - ``aggressive``: divide a cube into smaller subcubes if at least one element can be
    //!   used for at least one of the subcubes multipole expansions. This
    //!   option, like all others, respects the maximum_tree_depth.
    //! - ``balanced``
    //! - ``guided``
    KFastMultipoleMatrixGeneratorBuilder::Attribute<std::string>("strategy") +
    //! the world cube gets divided into this many cubes along each side
    //! to create the first level of cubes. For MPI the number of top level cubes
    //! (which is top_level_divisions^3) has to be at least the number of MPI nodes.
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("top_level_divisions") +
    //! in all but the first subdivision the cubes get divided into this many
    //! cubes along each side. Two is the smallest useful value and a very good choice, too.
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("tree_level_divisions") +
    //! the maximum degree of the multipole and local expansions used in the algorithm
    //! for precise results values in the range of 10 to 13 are recommended combined
    //! with a zero_mask size of 3. The example uses smaller values to speed up the computation
    //! to a manageable time for a demonstation.
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("expansion_degree") +
    //! the distance in cubes that neighbors are defined by
    //! Cubes being having less than this number of cubes inbetween them in any
    //! direction are considered neighbors.
    //! Neighbors define the dead zone where the multipole expansion is not valid.
    //! Greater numbers like 3 make the algorithm more accurate but slower
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("neighbor_order") +
    //! the maximum depth to which cubes are subdivided
    //! this restricts the tree size and computation time for the local moments
    //! at the cost of more direct evaluations later
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("maximum_tree_depth") +
    //! the factor by which the estimated region for the world cube is expanded
    //! in size to compensate for slight errors in the region estimation algorithm
    //! and to ensure all source elements are inside the world cube
    KFastMultipoleMatrixGeneratorBuilder::Attribute<double>("region_expansion_factor") +
    //! this boolean option allows the automatic selection of the world cube
    //! from the underlying geometry so that all field sources are covered.
    //! Use with a region expansion factor (for example 1.1) to compensate for slight
    //! errors in the estimate. Try larger ones if elements are missed out.
    KFastMultipoleMatrixGeneratorBuilder::Attribute<bool>("use_region_size_estimation") +
    //! if the region estimation in not used, the cube center can be defined
    //! manually by passing this option the vector of the user's choice
    KFastMultipoleMatrixGeneratorBuilder::Attribute<KEMStreamableThreeVector>("world_cube_center") +
    //! manually set the side length of the world cube
    KFastMultipoleMatrixGeneratorBuilder::Attribute<double>("world_cube_length") +
    //! currently without effect
    KFastMultipoleMatrixGeneratorBuilder::Attribute<bool>("use_caching") +
    //! values between 0 and 3 define the amount of output on the console.
    //! A value of zero is mostly silent and a value of three gives maximum
    //! information.
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("verbosity") +
    //! elements are accepted by a cube if they are within the smallest ball around the
    //! cube multiplied by this factor which should be greater than one (for example 1.3).
    //! This allows small elements sitting on the borders of cubes to be accepted by small
    //! cubes and limit direct evaluation to a more local region.
    KFastMultipoleMatrixGeneratorBuilder::Attribute<double>("insertion_ratio") +
    //! if strategy = Guided, number of elements above which subdivision is triggered
    KFastMultipoleMatrixGeneratorBuilder::Attribute<unsigned int>("allowed_number") +
    //! if strategy = Guided, fraction of elements abouve which subdivision is triggered
    KFastMultipoleMatrixGeneratorBuilder::Attribute<double>("allowed_fraction") +
    //! if strategy = Balanced, scale factor for biasing fft events
    KFastMultipoleMatrixGeneratorBuilder::Attribute<double>("bias_degree");

template<class Builder> int AddMatrixAsElement()
{
    return Builder::template ComplexElement<KFastMultipoleMatrixGenerator>("fast_multipole_matrix") +
           Builder::template ComplexElement<KFastMultipoleMatrixGenerator>("fftm_multiplication");
}

STATICINT sKKrylovChargeDensitySolverStructure = AddMatrixAsElement<KKrylovChargeDensitySolverBuilder>();

STATICINT sKKrylvoChargeDensitySolverStructure = AddMatrixAsElement<KKrylovPreconditionerGeneratorBuilder>();

} /* namespace katrin */

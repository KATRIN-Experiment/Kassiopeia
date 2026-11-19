/*
 * KElectricFastMultipoleFieldSolverBuilder.cc
 *
 *  Created on: 24 Jul 2015
 *      Author: wolfgang
 */

#include "KElectricFastMultipoleFieldSolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

using namespace KEMField;

namespace katrin
{

template<> KElectricFastMultipoleFieldSolverBuilder::~KComplexElement() = default;

STATICINT sKElectricFastMultipoleFieldSolverStructure =
    //! the world cube gets divided into this many cubes along each side
    //! to create the first level of cubes.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("top_level_divisions") +
    //! in all but the first subdivision the cubes get divided into this many
    //! cubes along each side. Two is the smallest useful value and a very good choice, too.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("tree_level_divisions") +
    //! the maximum degree of the multipole and local expansions used in the algorithm
    //! for precise results values in the range of 10 to 13 are recommended combined
    //! with a zero_mask size of 3. The example uses smaller values to speed up the computation
    //! to a manageable time for a demonstation.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("expansion_degree") +
    //! the distance in cubes that neighbors are defined by
    //! Cubes being having less than this number of cubes inbetween them in any
    //! direction are considered neighbors.
    //! Neighbors define the dead zone where the multipole expansion is not valid.
    //! Greater numbers like 3 make the algorithm more accurate but slower
    KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("neighbor_order") +
    //! the maximum depth to which cubes are subdivided
    //! this restricts the tree size and computation time for the local moments
    //! at the cost of more direct evaluations later
    KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("maximum_tree_depth") +
    //! the factor by which the estimated region for the world cube is expanded
    //! in size to compensate for slight errors in the region estimation algorithm
    //! and to ensure all source elements are inside the world cube
    KElectricFastMultipoleFieldSolverBuilder::Attribute<double>("region_expansion_factor") +
    //! this boolean option allows the automatic selection of the world cube
    //! from the underlying geometry so that all field sources are covered.
    //! Use with a region expansion factor (for example 1.1) to compensate for slight
    //! errors in the estimate. Try larger ones if elements are missed out.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<bool>("use_region_size_estimation") +
    KElectricFastMultipoleFieldSolverBuilder::Attribute<double>("world_cube_center_x") +
    KElectricFastMultipoleFieldSolverBuilder::Attribute<double>("world_cube_center_y") +
    //! if the region estimation in not used, the cube center can be defined
    //! manually by passing these options the position of the user's choice
    KElectricFastMultipoleFieldSolverBuilder::Attribute<double>("world_cube_center_z") +
    //! manually set the side length of the world cube
    KElectricFastMultipoleFieldSolverBuilder::Attribute<double>("world_cube_length") +
    //! currently without effect
    KElectricFastMultipoleFieldSolverBuilder::Attribute<bool>("use_caching") +
    //! values between 0 and 3 define the amount of output on the console.
    //! A value of zero is mostly silent and a value of three gives maximum
    //! information.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<unsigned int>("verbosity") +
    //! elements are accepted by a cube if they are within the smallest ball around the
    //! cube multiplied by this factor which should be greater than one (for example 1.3).
    //! This allows small elements sitting on the borders of cubes to be accepted by small
    //! cubes and limit direct evaluation to a more local region.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<double>("insertion_ratio") +
    //! use OpenCl to speed up the initialization process and direct evaluations.
    KElectricFastMultipoleFieldSolverBuilder::Attribute<bool>("use_opencl") +
    KElectricFastMultipoleFieldSolverBuilder::Attribute<bool>("split_mode");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KElectricFastMultipoleFieldSolver>(
        "fast_multipole_field_solver");

} /* namespace katrin */

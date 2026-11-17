/*
 * KRobinHoodChargeDensitySolverBuilder.cc
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#include "KRobinHoodChargeDensitySolverBuilder.hh"

#include "KElectrostaticBoundaryFieldBuilder.hh"

#include <string>

using namespace KEMField;

namespace katrin
{

template<> KRobinHoodChargeDensitySolverBuilder::~KComplexElement() = default;

STATICINT sKRobinHoodChargeDensitySolverStructure =
    //! specify the desired reduction of the start residual to define
    //! convergence. This defaults to 1e-8, meaning that the algorithm will
    //! stop when the norm of the residual has fallen to 1e-8 times its value
    //! in the first iteration (the norm of the vector describing the desired
    //! potential distribution).
    KRobinHoodChargeDensitySolverBuilder::Attribute<double>("tolerance") +
    //! the algorithm will iterate this many lines of the matrix before
    //! checking for convergence and allowing visitors to run.
    //! Defaults to the dimension of the boundary element matrix.
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("check_sub_interval") +
    //! specify how often the state of the iteration with number of
    //! iterations  and residual is displayed. While the value is integer it
    //! will display every check_sub_interval except for the case that the
    //! value is 0.
    //! It defaults to 0.
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("display_interval") +
    //! specify how often the state of the iteration is saved. It allows to
    //! recover the iteration from this state if the job has been
    //! interrupted. While the value is integer it will display every
    //! check_sub_interval except for the case that the value is 0.
    //! It defaults to 0.
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("write_interval") +
    //! display the convergence curve of the residuum in a graph
    //! While the value is integer it will display every check_sub_interval
    //! except for the case that the value is 0.
    //! It defaults to 0.
    KRobinHoodChargeDensitySolverBuilder::Attribute<unsigned int>("plot_interval") +
    //! boolean arguement to store the computed matrix elements of the
    //! boundary element matrix. This potentially speeds the computation up
    //! as elements don't have to be recomputed but it is only possible if
    //! enough RAM is available to store all matrix elements. The storage
    //! size scales with n^2, where n is the number of discretized boundary
    //! elements.
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("cache_matrix_elements") +
    //! speed up the computation by using OpenCL, only when compiled with
    //! OpenCL and an available parallel device (i.e graphics card).
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("use_opencl") +
    //! this option has to be true for the plot_interval parameter to create
    //! a graph showing the convergence
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("use_vtk") +
    KRobinHoodChargeDensitySolverBuilder::Attribute<bool>("split_mode") +
    //! @values numeric, analytic, rwg, reference, default
    KRobinHoodChargeDensitySolverBuilder::Attribute<std::string>("integrator");

STATICINT sKElectrostaticBoundaryField =
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KRobinHoodChargeDensitySolver>("robin_hood_bem_solver") +
    KElectrostaticBoundaryFieldBuilder::ComplexElement<KRobinHoodChargeDensitySolver>(
        "robin_hood_charge_density_solver");
} /* namespace katrin */

/*
 * KKrylovChargeDensitySolverBuilder.hh
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_
#define KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_

#include "KBoundaryMatrixGenerator.hh"
#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KEMStringUtils.hh"
#include "KKrylovChargeDensitySolver.hh"
#include "KKrylovSolverConfigurationReader.hh"


namespace katrin
{

//! the Krylov charge density solver is a solver using Krylov subspace methods
//! to solve the linear charge densities equation.
//! It offers the choice of GMRES and BiCGStab and requires to specify a matrix
//! vector multiplication method to convert charge densities into potentials
//! on the source elements. As of today only the fast multipole matrix is available.
//! Optionally a preconditioner can be added. For details on the matrix and preconditioners
//! see the next examples.
//! @example
//! <krylov_charge_density_solver
//!     solver_name="gmres"
//!     tolerance="1e-8"
//!     max_iterations="500"
//!     iterations_between_restarts="500"
//!     use_display="true"
//!     show_plot="true"
//! >
//!     <fast_multipole_matrix/>
//! </krylov_charge_density_solver>
typedef KComplexElement<KEMField::KKrylovChargeDensitySolver> KKrylovChargeDensitySolverBuilder;

template<> inline bool KKrylovChargeDensitySolverBuilder::AddAttribute(KContainer* aContainer)
{
    if (SetKrylovSolverConfiguration(*aContainer, *fObject))
        return true;
    return false;
}

template<> inline bool KKrylovChargeDensitySolverBuilder::AddElement(KContainer* anElement)
{
    if (SetKrylovSolverMatrixAndPrecon(*anElement, *fObject))
        return true;
    return false;
}

template<> inline bool KKrylovChargeDensitySolverBuilder::End()
{
    if (!fObject->GetMatrixGenerator()) {
        BINDINGMSG(eError) << " No matrix specified in krylov_charge_"
                              "density_solver or krylov_bem_solver."
                           << eom;
    }
    else
        return true;
    return false;
}

} /* namespace katrin */

#endif /* KKRYLOVCHARGEDENSITYSOLVERBUILDER_HH_ */

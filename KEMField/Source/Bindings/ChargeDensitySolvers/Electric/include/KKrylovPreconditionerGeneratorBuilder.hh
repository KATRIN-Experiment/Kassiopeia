/*
 * KKrylovPreconditionerGeneratorBuilder.hh
 *
 *  Created on: 19 Aug 2015
 *      Author: wolfgang
 */

#ifndef KKRYLOVPRECONDITIONERGENERATORBUILDER_HH_
#define KKRYLOVPRECONDITIONERGENERATORBUILDER_HH_

#include "KComplexElement.hh"
#include "KEMBindingsMessage.hh"
#include "KKrylovPreconditionerGenerator.hh"
#include "KKrylovSolverConfigurationReader.hh"

namespace katrin
{

//! the preconditioner is a inner Krylov solver as right preconditioner for Krylov solvers.
//! In each main iteration step the solution is multiplied with the preconditioner matrix first.
//! An approximate inverse of the matrix works best and is implemented as a fast and not too accurate
//! Krylov solver. Accordingly, the preconditioner offers the same parameters as the Krylov solver
//! and requires matrix too, in this case a less accurate but much faster choice is advised.
//! Furthermore the Krylov preconditioner can also accept an inner preconditioner, just like
//! the Krylov solver. This is called multilevel preconditioning.
//! @example
//! <krylov_preconditioner
//!     solver_name="gmres"
//!     tolerance="1e-2"
//!     max_iterations="5"
//!     iterations_between_restarts="5"
//!     use_display="true"
//!     show_plot="true"
//! >
//!     <fast_multipole_matrix
//!         top_level_divisions="4"
//!         tree_level_divisions="2"
//!         expansion_degree="1"
//!         neighbor_order="2"
//!         maximum_tree_depth="2"
//!         region_expansion_factor="1.1"
//!         use_region_size_estimation="true"
//!         verbosity="3"
//!     />
//! </krylov_preconditioner>
typedef KComplexElement<KEMField::KKrylovPreconditionerGenerator> KKrylovPreconditionerGeneratorBuilder;

template<> inline bool KKrylovPreconditionerGeneratorBuilder::AddAttribute(KContainer* aContainer)
{
    if (SetKrylovSolverConfiguration(*aContainer, *fObject))
        return true;
    return false;
}

template<> inline bool KKrylovPreconditionerGeneratorBuilder::AddElement(KContainer* anElement)
{
    if (SetKrylovSolverMatrixAndPrecon(*anElement, *fObject))
        return true;
    return false;
}

template<> inline bool KKrylovPreconditionerGeneratorBuilder::End()
{
    if (!fObject->GetMatrixGenerator()) {
        BINDINGMSG(eError) << " No matrix specified in krylov_preconditioner." << eom;
    }
    else
        return true;
    return false;
}

} /* namespace katrin */

#endif /* KKRYLOVPRECONDITIONERGENERATORBUILDER_HH_ */

/*
 * KElectrostaticIntegratingFieldSolverTemplate.hh
 *
 *  Created on: 2 Jul 2015
 *      Author: wolfgang
 */

#ifndef KINTEGRATINGFIELDSOLVERTEMPLATE_HH_
#define KINTEGRATINGFIELDSOLVERTEMPLATE_HH_

namespace KEMField {

/**
 *  This partial specialisation rule allows to use different boundary integrators for one
 *  type of integrating field solver. By defining the Kind type of the integrator
 *  the corresponding KIntegratingFieldSolver template is selected.
 *  This way a boundary integrator can state whether it is ElectrostaticSingleThread,
 *  ElectrostaticOpenCL or ElectromagnetSingleThread (no OpenCL version here).
 *  The KIntegratingFieldSolver will just need one template argument and figure out
 *  with the help of the boundary integrator which template specialisation is adequat.
 */
template <class Integrator, typename Kind = typename Integrator::Kind>
class KIntegratingFieldSolver;

}



#endif /* KINTEGRATINGFIELDSOLVERTEMPLATE_HH_ */

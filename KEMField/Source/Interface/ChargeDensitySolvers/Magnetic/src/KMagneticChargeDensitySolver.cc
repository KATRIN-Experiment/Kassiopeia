/*
 * KMagneticChargeDensitySolver.cc
 *
 *  Created on: 2 Apr 2025
 *      Author: pslocum
 */

#include "KMagneticChargeDensitySolver.hh"

#ifdef KEMFIELD_USE_ROOT
#endif

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (KEMField::KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_SINGLE_PROCESS
#define MPI_SINGLE_PROCESS if (true)
#endif
#endif

using namespace std;

namespace KEMField
{
    bool KMagneticChargeDensitySolver::FindSolution(double threshold, KSurfaceContainer& container)
    {
        if (container.empty())
        {
            kem_cout(eError) << "ERROR: Solver got no electrode elements (did you forget to setup a geometry mesh?)" << eom;
        }
        if (threshold < 0.)
        {
            kem_cout(eError) << "ERROR: Threshold is < 0.)" << eom;            
		}        
        return true;
    }


}  // namespace KEMField

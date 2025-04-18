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
    bool KMagneticChargeDensitySolver::FindSolution()
    {
        return true;
    }


}  // namespace KEMField

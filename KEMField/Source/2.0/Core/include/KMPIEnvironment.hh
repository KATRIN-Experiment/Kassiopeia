/*
 * MPIEnvironment.hh
 *
 * Includes MPI and defines common macros.
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_CORE_INCLUDE_KMPIENVIRONMENT_HH_
#define KEMFIELD_SOURCE_2_0_CORE_INCLUDE_KMPIENVIRONMENT_HH_

#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
#endif

#ifdef KEMFIELD_USE_MPI
    #define MPI_SINGLE_PROCESS if ( KEMField::KMPIInterface::GetInstance()->GetProcess()==0 )
    #define MPI_SECOND_PROCESS if ( KEMField::KMPIInterface::GetInstance()->GetProcess()==1 )
#else
    #define MPI_SINGLE_PROCESS
    #define MPI_SECOND_PROCESS
#endif

#endif /* KEMFIELD_SOURCE_2_0_CORE_INCLUDE_KMPIENVIRONMENT_HH_ */

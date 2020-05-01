/*
 * KKrylovSolverConfiguration.cc
 *
 *  Created on: 12 Aug 2015
 *      Author: wolfgang
 */

#include "KKrylovSolverConfiguration.hh"

#include <climits>
namespace KEMField
{

KKrylovSolverConfiguration::KKrylovSolverConfiguration() :
    fSolverName("gmres"),
    fTolerance(0.1),
    fMaxIterations(UINT_MAX),
    fIterationsBetweenRestart(UINT_MAX),
    fUseCheckpoints(false),
    fStepsBetweenCheckpoints(1),
    fUseDisplay(false),
    fUsePlot(false),
    fUseTimer(false),
    fTimeLimitSeconds(3e10),  //seconds
    fStepsBetweenTimeChecks(1)

{}

KKrylovSolverConfiguration::~KKrylovSolverConfiguration() {}

} /* namespace KEMField */

/*
 * KElectrostaticBoundaryFieldTimer.cc
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#include "KElectrostaticBoundaryFieldTimer.hh"

namespace KEMField
{

KElectrostaticBoundaryFieldTimer::KElectrostaticBoundaryFieldTimer() :
    fChargeDensityTimer("charge density solver initialization"),
    fFieldSolverTimer("field solver initialization")
{
    Preprocessing(true);
    InBetweenProcessing(true);
    Postprocessing(true);
}

KElectrostaticBoundaryFieldTimer::~KElectrostaticBoundaryFieldTimer() = default;

void KElectrostaticBoundaryFieldTimer::PreVisit(KElectrostaticBoundaryField& /*unused*/)
{
    fChargeDensityTimer = KTimer("charge density solver initialization");
    fChargeDensityTimer.start();
}

void KElectrostaticBoundaryFieldTimer::InBetweenVisit(KElectrostaticBoundaryField& /*unused*/)
{
    fChargeDensityTimer.end();
    fChargeDensityTimer.display();
    fFieldSolverTimer = KTimer("field solver initialization");
    fFieldSolverTimer.start();
}

void KElectrostaticBoundaryFieldTimer::PostVisit(KElectrostaticBoundaryField& /*unused*/)
{
    fFieldSolverTimer.end();
    fFieldSolverTimer.display();
}

} /* namespace KEMField */

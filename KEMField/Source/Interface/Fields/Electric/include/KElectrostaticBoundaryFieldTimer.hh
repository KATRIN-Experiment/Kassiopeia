/*
 * KBoundaryFieldTimer.hh
 *
 *  Created on: 24 Sep 2015
 *      Author: wolfgang
 */

#ifndef KELECTROSTATICBOUNDARYFIELDTIMER_H_
#define KELECTROSTATICBOUNDARYFIELDTIMER_H_

#include "KElectrostaticBoundaryField.hh"
#include "KTimer.hh"

namespace KEMField
{

class KElectrostaticBoundaryFieldTimer : public KElectrostaticBoundaryField::Visitor
{
  public:
    KElectrostaticBoundaryFieldTimer();
    ~KElectrostaticBoundaryFieldTimer() override;

    void PreVisit(KElectrostaticBoundaryField&) override;
    void InBetweenVisit(KElectrostaticBoundaryField&) override;
    void PostVisit(KElectrostaticBoundaryField&) override;

  private:
    KTimer fChargeDensityTimer;
    KTimer fFieldSolverTimer;
};

} /* namespace KEMField */

#endif /* KELECTROSTATICBOUNDARYFIELDTIMER_H_ */

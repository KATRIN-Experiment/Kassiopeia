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

namespace KEMField {

class KElectrostaticBoundaryFieldTimer : public KElectrostaticBoundaryField::Visitor {
public:
    KElectrostaticBoundaryFieldTimer();
    virtual ~KElectrostaticBoundaryFieldTimer();

    virtual void PreVisit( KElectrostaticBoundaryField& );
    virtual void InBetweenVisit( KElectrostaticBoundaryField& );
    virtual void PostVisit( KElectrostaticBoundaryField& );

private:
    KTimer fChargeDensityTimer;
    KTimer fFieldSolverTimer;
};

} /* namespace KEMField */

#endif /* KELECTROSTATICBOUNDARYFIELDTIMER_H_ */

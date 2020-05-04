/*
 * KIterativeSolverTimer.hh
 *
 *  Created on: 17 Aug 2015
 *      Author: wolfgang
 */

#ifndef KITERATIVESOLVERTIMER_HH_
#define KITERATIVESOLVERTIMER_HH_

#include "KFMMessaging.hh"
#include "KIterativeSolver.hh"
#include "KTimer.hh"

namespace KEMField
{


template<typename ValueType> class KIterativeSolverTimer : public KIterativeSolver<ValueType>::Visitor
{
  public:
    KIterativeSolverTimer() : fTimer(sDescription) {}
    ~KIterativeSolverTimer() override {}

    void Initialize(KIterativeSolver<ValueType>&) override;
    void Visit(KIterativeSolver<ValueType>&) override {}
    void Finalize(KIterativeSolver<ValueType>&) override;

  private:
    KTimer fTimer;
    static const std::string sDescription;
};

template<typename ValueType> const std::string KIterativeSolverTimer<ValueType>::sDescription = "Iterative solver";

template<typename ValueType> void KIterativeSolverTimer<ValueType>::Initialize(KIterativeSolver<ValueType>&)
{
    fTimer.start();
}

template<typename ValueType> void KIterativeSolverTimer<ValueType>::Finalize(KIterativeSolver<ValueType>&)
{
    fTimer.end();
    fTimer.display();
}

}  // namespace KEMField

#endif /* KITERATIVESOLVERTIMER_HH_ */

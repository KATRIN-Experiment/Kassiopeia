/*
 * KBiotSavartChargeDensitySolver.hh
 *
 *  Created on: 4 Apr 2025
 *      Author: pslocum
 */

#ifndef KBIOTSAVARTCHARGEDENSITYSOLVER_HH_
#define KBIOTSAVARTCHARGEDENSITYSOLVER_HH_

#include "KMagneticChargeDensitySolver.hh"

namespace KEMField
{

template<typename ValueType> class KBoundaryMatrixGenerator;  // forward declaration

class KBiotSavartChargeDensitySolver : public KMagneticChargeDensitySolver
{
  public:

    KBiotSavartChargeDensitySolver();
    ~KBiotSavartChargeDensitySolver() override;

  private:

    void InitializeCore(KSurfaceContainer& container) override;



};

} /* namespace KEMField */

#endif /* KBIOTSAVARTCHARGEDENSITYSOLVER_HH_ */

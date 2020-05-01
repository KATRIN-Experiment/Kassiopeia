/*
 * KZonalHarmonicMagnetostaticFieldSolver.hh
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#ifndef KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_
#define KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_

#include "KElectromagnetZonalHarmonicFieldSolver.hh"
#include "KMagneticFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"

namespace KEMField
{

class KZonalHarmonicMagnetostaticFieldSolver : public KMagneticFieldSolver
{
  public:
    KZonalHarmonicMagnetostaticFieldSolver();
    ~KZonalHarmonicMagnetostaticFieldSolver() override;

    void InitializeCore(KElectromagnetContainer& container) override;

    KThreeVector MagneticPotentialCore(const KPosition& P) const override;
    KThreeVector MagneticFieldCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;
    std::pair<KThreeVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const override;

    KZonalHarmonicParameters* GetParameters()
    {
        return fParameters;
    }

  private:
    KElectromagnetIntegrator fIntegrator;
    KZonalHarmonicContainer<KMagnetostaticBasis>* fZHContainer;
    KZonalHarmonicFieldSolver<KMagnetostaticBasis>* fZonalHarmonicFieldSolver;
    KZonalHarmonicParameters* fParameters;
};

} /* namespace KEMField */

#endif /* KZONALHARMONICMAGNETOSTATICFIELDSOLVER_HH_ */

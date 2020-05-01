/*
 * KElectricZHFieldSolver.hh
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#ifndef KELECTRICZHFIELDSOLVER_HH_
#define KELECTRICZHFIELDSOLVER_HH_

#include "KElectricFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KElectrostaticZonalHarmonicFieldSolver.hh"
#include "KZonalHarmonicContainer.hh"
#include "KZonalHarmonicParameters.hh"

namespace KEMField
{

class KElectricZHFieldSolver : public KElectricFieldSolver
{
  public:
    KElectricZHFieldSolver();
    ~KElectricZHFieldSolver() override;

    bool UseCentralExpansion(const KPosition& P);
    bool UseRemoteExpansion(const KPosition& P);

    void SetIntegratorPolicy(const KEBIPolicy& policy)
    {
        fIntegratorPolicy = policy;
    }

    KZonalHarmonicParameters* GetParameters()
    {
        return fParameters;
    }

  private:
    void InitializeCore(KSurfaceContainer& container) override;

    double PotentialCore(const KPosition& P) const override;
    KThreeVector ElectricFieldCore(const KPosition& P) const override;
    std::pair<KThreeVector, double> ElectricFieldAndPotentialCore(const KPosition& P) const override;

    KEBIPolicy fIntegratorPolicy;
    KElectrostaticBoundaryIntegrator fIntegrator;
    KZonalHarmonicContainer<KElectrostaticBasis>* fZHContainer;
    KZonalHarmonicFieldSolver<KElectrostaticBasis>* fZonalHarmonicFieldSolver;
    KZonalHarmonicParameters* fParameters;
};

} /* namespace KEMField */

#endif /* KELECTRICZHFIELDSOLVER_HH_ */

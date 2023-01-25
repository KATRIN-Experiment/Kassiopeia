/*
 * KIntegratingElectrostaticFieldSolver.hh
 *
 *  Created on: 05.06.2015
 *      Author: gosda
 *
 *      This is just a wrapper class for bindings or other high level uses.
 *      Imported from KSFieldElectrostatic
 */

#ifndef KINTEGRATINGELECTROSTATICFIELDSOLVER_HH_
#define KINTEGRATINGELECTROSTATICFIELDSOLVER_HH_

#include "KElectricFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KElectrostaticIntegratingFieldSolver.hh"
#ifdef KEMFIELD_USE_OPENCL
#include "KOpenCLElectrostaticIntegratingFieldSolver.hh"
#endif

namespace KEMField
{

class KIntegratingElectrostaticFieldSolver : public KElectricFieldSolver
{
  public:
    KIntegratingElectrostaticFieldSolver();
    ~KIntegratingElectrostaticFieldSolver() override;

    void SetIntegratorPolicy(KEBIPolicy& policy)
    {
        fIntegratorPolicy = policy;
    }

    void UseOpenCL(bool choice)
    {
        fUseOpenCL = choice;
    }

  private:
    void InitializeCore(KSurfaceContainer& container) override;
    void DeinitializeCore() override {}

    double PotentialCore(const KPosition& P) const override;
    KFieldVector ElectricFieldCore(const KPosition& P) const override;

    KElectrostaticBoundaryIntegrator* fIntegrator;
    KEBIPolicy fIntegratorPolicy;
    KIntegratingFieldSolver<KElectrostaticBoundaryIntegrator>* fIntegratingFieldSolver;

#ifdef KEMFIELD_USE_OPENCL
    KOpenCLElectrostaticBoundaryIntegrator* fOCLIntegrator;
    KIntegratingFieldSolver<KOpenCLElectrostaticBoundaryIntegrator>* fOCLIntegratingFieldSolver;
#endif

    bool fUseOpenCL;
};

}  // namespace KEMField

#endif /* KINTEGRATINGELECTROSTATICFIELDSOLVER_HH_ */

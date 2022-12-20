/*
 * KGaussSeidelChargeDensitySolver.hh
 *
 *  Created on: 18 Jun 2015
 *      Author: wolfgang
 */

#ifndef KGAUSSSEIDELCHARGEDENSITYSOLVER_HH_
#define KGAUSSSEIDELCHARGEDENSITYSOLVER_HH_

#include "KChargeDensitySolver.hh"
#include "KEMCoreMessage.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"

namespace KEMField
{

class KGaussSeidelChargeDensitySolver : public KChargeDensitySolver
{
  public:
    KGaussSeidelChargeDensitySolver();
    ~KGaussSeidelChargeDensitySolver() override;

    void InitializeCore(KSurfaceContainer& container) override;

    void SetIntegratorPolicy(const KEBIPolicy& policy)
    {
        fIntegratorPolicy = policy;
    }
    void UseOpenCL(bool choice)
    {
        if (choice == true) {
#ifdef KEMFIELD_USE_OPENCL
            fUseOpenCL = choice;
            return;
#endif
            kem_cout(eWarning) << "WARNING: cannot use OpenCl in robin hood without"
                                  " KEMField being built with OpenCl, using defaults."
                               << eom;
        }
        fUseOpenCL = false;
        return;
    }

  private:
    KEBIPolicy fIntegratorPolicy;
    bool fUseOpenCL;
};

}  // namespace KEMField

#endif /* KGAUSSSEIDELCHARGEDENSITYSOLVER_HH_ */

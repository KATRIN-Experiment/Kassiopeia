/*
 * KRobinHoodChargeDensitySolver.hh
 *
 *  Created on: 29 Jul 2015
 *      Author: wolfgang
 */

#ifndef KROBINHOODCHARGEDENSITYSOLVER_HH_
#define KROBINHOODCHARGEDENSITYSOLVER_HH_

#include "KChargeDensitySolver.hh"
#include "KEMCoreMessage.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"

namespace KEMField
{

class KRobinHoodChargeDensitySolver : public KChargeDensitySolver
{
  public:
    KRobinHoodChargeDensitySolver();
    ~KRobinHoodChargeDensitySolver() override;

    void SetIntegratorPolicy(const KEBIPolicy& policy)
    {
        fIntegratorPolicy = policy;
    }
    void SetTolerance(double d)
    {
        fTolerance = d;
    }
    void SetCheckSubInterval(unsigned int i)
    {
        fCheckSubInterval = i;
    }
    void SetDisplayInterval(unsigned int i)
    {
        fDisplayInterval = i;
    }
    void SetWriteInterval(unsigned int i)
    {
        fWriteInterval = i;
    }
    void SetPlotInterval(unsigned int i)
    {
        fPlotInterval = i;
    }
    void CacheMatrixElements(bool choice)
    {
        fCacheMatrixElements = choice;
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
    void UseVTK(bool choice)
    {
        if (choice == true) {
#ifdef KEMFIELD_USE_VTK
            fUseVTK = choice;
            return;
#endif
            kem_cout(eWarning) << "WARNING: cannot use vtk in robin hood without"
                                  " KEMField being built with vtk, using defaults."
                               << eom;
        }
        fUseVTK = false;
        return;
    }
    void SetSplitMode(bool choice);

  private:
    void InitializeCore(KSurfaceContainer& container) override;

    KEBIPolicy fIntegratorPolicy;
    double fTolerance;
    unsigned int fCheckSubInterval;
    unsigned int fDisplayInterval;
    unsigned int fWriteInterval;
    unsigned int fPlotInterval;
    bool fCacheMatrixElements;
    bool fUseOpenCL;
    bool fUseVTK;
};

}  // namespace KEMField

#endif /* KROBINHOODCHARGEDENSITYSOLVER_HH_ */

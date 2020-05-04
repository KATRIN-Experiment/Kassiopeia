/*
 * KElectricFastMultipoleFieldSolver.hh
 *
 *  Created on: 23.07.2015
 *      Author: gosda
 */

#ifndef KELECTRICFASTMULTIPOLEFIELDSOLVER_HH_
#define KELECTRICFASTMULTIPOLEFIELDSOLVER_HH_

#include "KElectricFieldSolver.hh"
#include "KElectrostaticBoundaryIntegratorPolicy.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver.hh"
#include "KFMElectrostaticParameters.hh"
#include "KFMElectrostaticTree.hh"

#ifdef KEMFIELD_USE_OPENCL
//#include "KFMElectrostaticFieldMapper_OpenCL.hh"
#include "KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh"
#endif

namespace KEMField
{

class KElectricFastMultipoleFieldSolver : public KElectricFieldSolver
{
  public:
    KElectricFastMultipoleFieldSolver();
    ~KElectricFastMultipoleFieldSolver() override;

    void InitializeCore(KSurfaceContainer& container) override;

    double PotentialCore(const KPosition& P) const override;
    KThreeVector ElectricFieldCore(const KPosition& P) const override;

    void SetIntegratorPolicy(const KEBIPolicy& policy)
    {
        fIntegratorPolicy = policy;
    }

    KFMElectrostaticParameters* GetParameters()
    {
        return &fParameters;
    }

    void UseOpenCL(bool choice);

  private:
    KEBIPolicy fIntegratorPolicy;
    KFMElectrostaticParameters fParameters;
    KFMElectrostaticTree* fTree;
    KFMElectrostaticFastMultipoleFieldSolver* fFastMultipoleFieldSolver;
#ifdef KEMFIELD_USE_OPENCL
    KFMElectrostaticFastMultipoleFieldSolver_OpenCL* fFastMultipoleFieldSolverOpenCL;
#endif
    bool fUseOpenCL;
};

} /* namespace KEMField */

#endif /* KELECTRICFASTMULTIPOLEFIELDSOLVER_HH_ */

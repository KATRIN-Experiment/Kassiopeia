/*
 * KMagfieldCoilsFieldSolver.hh
 *
 *  Created on: 31 Jan 2023
 *      Author: Jan Behrens
 */

#ifndef KMagfieldCoilsFieldSolver_HH_
#define KMagfieldCoilsFieldSolver_HH_

#include "KMagneticFieldSolver.hh"
#include "MagfieldCoils.h"

#include <memory>

namespace katrin
{
class KTextFile;
};

namespace KEMField
{

class KMagfieldCoilsFieldSolver : public KMagneticFieldSolver
{

  public:
    KMagfieldCoilsFieldSolver();

    void SetDirName(const std::string& aName) { fDirName = aName; }
    void SetObjectName(const std::string& aName) { fObjectName = aName; }
    void SetCoilFileName(const std::string& aName) { fCoilFileName = aName; }
    void SetReplaceFile(bool aFlag) { fReplaceFile = aFlag; }
    void SetForceElliptic(bool aFlag) { fForceElliptic = aFlag; }
    void SetNElliptic(int aNumber) { fNElliptic = aNumber; }
    void SetNMax(int aNumber) { fNMax = aNumber; }
    void SetEpsTol(double aNumber) { fEpsTol = aNumber; }

  private:
    void InitializeCore(KElectromagnetContainer& aContainer) override;
    void DeinitializeCore() override;

    KFieldVector MagneticPotentialCore(const KPosition& P) const override;
    KFieldVector MagneticFieldCore(const KPosition& P) const override;
    KGradient MagneticGradientCore(const KPosition& P) const override;

    std::string WriteCoilFile(katrin::KTextFile* aFile, KElectromagnetContainer& aContainer);

  private:
    std::shared_ptr<MagfieldCoils> fSolver;

    std::string fDirName;
    std::string fObjectName;
    std::string fCoilFileName;
    bool fReplaceFile;
    bool fForceElliptic;
    int fNElliptic;
    int fNMax;
    double fEpsTol;

    mutable unsigned int fApproxExecCount, fDirectExecCount;
};

} /* namespace KEMField */

#endif /* KMagfieldCoilsFieldSolver_HH_ */

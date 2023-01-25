/*
 * KStaticElectroMagnetField.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KSTATICELECTROMAGNETFIELD_HH_
#define KSTATICELECTROMAGNETFIELD_HH_

#include "KMagneticFieldSolver.hh"
#include "KMagnetostaticField.hh"

#include <memory>

namespace KEMField
{

class KStaticElectromagnetField : public KMagnetostaticField
{
  public:
    KStaticElectromagnetField();
    ~KStaticElectromagnetField() override;

    void SetDirectory(const std::string& aDirectory);
    std::string GetDirectory() const;

    void SetFile(const std::string& aFile);
    std::string GetFile() const;

    void SetFieldSolver(const std::shared_ptr<KMagneticFieldSolver>& solver);
    std::shared_ptr<KMagneticFieldSolver> GetFieldSolver();

    void SetContainer(const std::shared_ptr<KElectromagnetContainer>& aContainer);
    std::shared_ptr<KElectromagnetContainer> GetContainer() const;

  protected:
    void InitializeCore() override;
    void DeinitializeCore() override;
    void CheckSolverExistance() const;

    KFieldVector MagneticPotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector MagneticFieldCore(const KPosition& aSamplePoint) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const override;
    std::pair<KFieldVector, KGradient> MagneticFieldAndGradientCore(const KPosition& P) const override;

  private:
    std::shared_ptr<KElectromagnetContainer> fContainer;
    std::shared_ptr<KMagneticFieldSolver> fFieldSolver;

    std::string fFile;
    std::string fDirectory;
};

} /* namespace KEMField */

#endif /* KSTATICELECTROMAGNETFIELD_HH_ */

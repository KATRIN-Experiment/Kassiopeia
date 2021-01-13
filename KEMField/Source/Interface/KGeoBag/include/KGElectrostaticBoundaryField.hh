/*
 * KGElectrostaticBoundaryField.hh
 *
 *  Created on: 15 Jun 2015
 *      Author: wolfgang
 */

#ifndef KGELECTROSTATICBOUNDARYFIELD_HH_
#define KGELECTROSTATICBOUNDARYFIELD_HH_

#include "KElectrostaticBoundaryField.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGCore.hh"

#include <string>
#include <vector>


namespace KEMField
{

class KGElectrostaticBoundaryField : public KElectrostaticBoundaryField
{
  public:
    KGElectrostaticBoundaryField();
    ~KGElectrostaticBoundaryField() override;
    enum Symmetry
    {
        NoSymmetry,
        AxialSymmetry,
        DiscreteAxialSymmetry
    };

    void SetMinimumElementArea(const double& aArea);
    void SetMaximumElementAspectRatio(const double& aAspect);

    void SetSystem(KGeoBag::KGSpace* aSpace);
    void AddSurface(KGeoBag::KGSurface* aSurface);
    void AddSpace(KGeoBag::KGSpace* aSpace);
    void SetSymmetry(const Symmetry& aSymmetry);
    KSmartPointer<KGeoBag::KGBEMConverter> GetConverter();


  private:
    double PotentialCore(const KPosition& P) const override;
    KFieldVector ElectricFieldCore(const KPosition& P) const override;
    void InitializeCore() override;

    void ConfigureSurfaceContainer();

    double fMinimumElementArea;
    double fMaximumElementAspectRatio;
    KGeoBag::KGSpace* fSystem;
    std::vector<KGeoBag::KGSurface*> fSurfaces;
    std::vector<KGeoBag::KGSpace*> fSpaces;
    Symmetry fSymmetry;

  private:
    KSmartPointer<KGeoBag::KGBEMConverter> fConverter;
};

}  // namespace KEMField


#endif /* KGELECTROSTATICBOUNDARYFIELD_HH_ */

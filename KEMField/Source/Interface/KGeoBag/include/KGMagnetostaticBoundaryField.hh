/*
 * KGMagnetostaticBoundaryField.hh
 *
 *  Created on: 07 May 2025
 *      Author: pslocum
 */

#ifndef KGMAGNETOSTATICBOUNDARYFIELD_HH_
#define KGMAGNETOSTATICBOUNDARYFIELD_HH_

#include "KMagnetostaticBoundaryField.hh"
#include "KGBEM.hh"
#include "KGBEMConverter.hh"
#include "KGCore.hh"

#include <string>
#include <vector>


namespace KEMField
{

class KGMagnetostaticBoundaryField : public KMagnetostaticBoundaryField
{
  public:
    KGMagnetostaticBoundaryField();
    ~KGMagnetostaticBoundaryField() override;
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
    std::shared_ptr<KGeoBag::KGBEMConverter> GetConverter();

    const std::vector<KGeoBag::KGSurface*>& GetSurfaces() const { return fSurfaces; }
    const std::vector<KGeoBag::KGSpace*>& GetSpaces() const { return fSpaces; }

  private:
    KFieldVector MagneticFieldCore(const KPosition& P) const override;
    void InitializeCore() override;

    void ConfigureSurfaceContainer();

    double fMinimumElementArea;
    double fMaximumElementAspectRatio;
    KGeoBag::KGSpace* fSystem;
    std::vector<KGeoBag::KGSurface*> fSurfaces;
    std::vector<KGeoBag::KGSpace*> fSpaces;
    Symmetry fSymmetry;

  private:
    std::shared_ptr<KGeoBag::KGBEMConverter> fConverter;
};

}  // namespace KEMField


#endif /* KGMAGNETOSTATICBOUNDARYFIELD_HH_ */

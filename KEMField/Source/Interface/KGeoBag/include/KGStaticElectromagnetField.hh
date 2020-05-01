/*
 * KGElectrostaticBoundaryField.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KGSTATICELECTROMAGNETFIELD_HH_
#define KGSTATICELECTROMAGNETFIELD_HH_

#include "KGCore.hh"
#include "KGElectromagnetConverter.hh"
#include "KStaticElectromagnetField.hh"

#include <string>
#include <vector>

namespace KEMField
{

class KGStaticElectromagnetField : public KStaticElectromagnetField
{
  public:
    KGStaticElectromagnetField();
    ~KGStaticElectromagnetField() override;

    void SetSystem(KGeoBag::KGSpace* aSystem);
    void AddSurface(KGeoBag::KGSurface* aSurface);
    void AddSpace(KGeoBag::KGSpace* aSpace);

    KSmartPointer<KGeoBag::KGElectromagnetConverter> GetConverter();

  private:
    void InitializeCore() override;

    KThreeVector MagneticPotentialCore(const KPosition& aSamplePoint) const override;
    KThreeVector MagneticFieldCore(const KPosition& aSamplePoint) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const override;

    void ConfigureSurfaceContainer();

    KGeoBag::KGSpace* fSystem;
    std::vector<KGeoBag::KGSurface*> fSurfaces;
    std::vector<KGeoBag::KGSpace*> fSpaces;

    KSmartPointer<KGeoBag::KGElectromagnetConverter> fConverter;
};

} /* namespace KEMField */

#endif /* KGSTATICELECTROMAGNETFIELD_HH_ */

/*
 * KGStaticElectromagnetField.hh
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

    void SetSaveMagfield3(bool aFlag);
    void SetDirectoryMagfield3(const std::string& aDirectory);

    std::shared_ptr<KGeoBag::KGElectromagnetConverter> GetConverter();

    const std::vector<KGeoBag::KGSurface*>& GetSurfaces() const { return fSurfaces; }
    const std::vector<KGeoBag::KGSpace*>& GetSpaces() const { return fSpaces; }

  private:
    void InitializeCore() override;

    KFieldVector MagneticPotentialCore(const KPosition& aSamplePoint) const override;
    KFieldVector MagneticFieldCore(const KPosition& aSamplePoint) const override;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const override;

    void ConfigureSurfaceContainer();

    KGeoBag::KGSpace* fSystem;
    std::vector<KGeoBag::KGSurface*> fSurfaces;
    std::vector<KGeoBag::KGSpace*> fSpaces;

    std::shared_ptr<KGeoBag::KGElectromagnetConverter> fConverter;

    bool fSaveMagfield3;
    std::string fDirectoryMagfield3;
};

} /* namespace KEMField */

#endif /* KGSTATICELECTROMAGNETFIELD_HH_ */

/*
 * KGElectrostaticBoundaryField.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KGSTATICELECTROMAGNETFIELD_HH_
#define KGSTATICELECTROMAGNETFIELD_HH_

#include "KStaticElectromagnetField.hh"
#include <string>
#include <vector>

#include "KGElectromagnetConverter.hh"
#include "KGCore.hh"

namespace KEMField {

class KGStaticElectromagnetField: public KStaticElectromagnetField {
public:
    KGStaticElectromagnetField();
    virtual ~KGStaticElectromagnetField();

    void SetSystem( KGeoBag::KGSpace* aSystem );
    void AddSurface( KGeoBag::KGSurface* aSurface );
    void AddSpace( KGeoBag::KGSpace* aSpace );

    KSmartPointer<KGeoBag::KGElectromagnetConverter> GetConverter();

private:
    void InitializeCore();

    KThreeVector MagneticPotentialCore(const KPosition& aSamplePoint) const;
    KThreeVector MagneticFieldCore(const KPosition& aSamplePoint) const;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const;

    void ConfigureSurfaceContainer();

    KGeoBag::KGSpace* fSystem;
    std::vector< KGeoBag::KGSurface* > fSurfaces;
    std::vector< KGeoBag::KGSpace* > fSpaces;

    KSmartPointer<KGeoBag::KGElectromagnetConverter> fConverter;
};

} /* namespace KEMField */

#endif /* KGSTATICELECTROMAGNETFIELD_HH_ */

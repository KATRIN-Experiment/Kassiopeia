/*
 * KStaticElectromagnetFieldWithKGeoBag.hh
 *
 *  Created on: 25 Mar 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_PLUGINS_KGEOBAG_INCLUDE_KSTATICELECTROMAGNETFIELDWITHKGEOBAG_HH_
#define KEMFIELD_SOURCE_2_0_PLUGINS_KGEOBAG_INCLUDE_KSTATICELECTROMAGNETFIELDWITHKGEOBAG_HH_

#include "KStaticElectromagnetField.hh"
#include <string>
#include <vector>
#include "KGCore.hh"
#include "KGElectromagnetConverter.hh"

namespace KEMField {

class KStaticElectromagnetFieldWithKGeoBag: public KStaticElectromagnetField {
public:
    KStaticElectromagnetFieldWithKGeoBag();
    virtual ~KStaticElectromagnetFieldWithKGeoBag();

    void SetSystem( KGeoBag::KGSpace* aSystem );
    void AddSurface( KGeoBag::KGSurface* aSurface );
    void AddSpace( KGeoBag::KGSpace* aSpace );

    KSmartPointer<KGeoBag::KGElectromagnetConverter> GetConverter();

private:
    void InitializeCore();

    KEMThreeVector MagneticPotentialCore(const KPosition& aSamplePoint) const;
    KEMThreeVector MagneticFieldCore(const KPosition& aSamplePoint) const;
    KGradient MagneticGradientCore(const KPosition& aSamplePoint) const;

    void ConfigureSurfaceContainer();

    KGeoBag::KGSpace* fSystem;
    std::vector< KGeoBag::KGSurface* > fSurfaces;
    std::vector< KGeoBag::KGSpace* > fSpaces;

    KSmartPointer<KGeoBag::KGElectromagnetConverter> fConverter;
};

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_PLUGINS_KGEOBAG_INCLUDE_KSTATICELECTROMAGNETFIELDWITHKGEOBAG_HH_ */

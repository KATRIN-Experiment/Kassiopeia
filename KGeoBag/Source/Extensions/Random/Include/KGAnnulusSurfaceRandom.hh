/*
 * KGAnnulusSurface.hh
 *
 *  Created on: 08.10.2020
 *      Author: J. Behrens
 */

#ifndef KGANNULUSSURFACERANDOM_HH_
#define KGANNULUSSURFACERANDOM_HH_

#include "KGAnnulusSurface.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
     * \brief Class for dicing a point on a KGAnnulusSurface.
     */
class KGAnnulusSurfaceRandom : virtual public KGShapeRandom, public KGAnnulusSurface::Visitor
{
  public:
    KGAnnulusSurfaceRandom() : KGShapeRandom() {}
    ~KGAnnulusSurfaceRandom() override = default;

    /**
         * \brief Visitor function to dice the point on the KGAnnulusSurface.
         *
         * \param anAnnulusSurface
         */
    void VisitAnnulusSurface(KGAnnulusSurface* anAnnulusSurface) override;
};
}  // namespace KGeoBag

#endif /* KGANNULUSSURFACERANDOM_HH_ */

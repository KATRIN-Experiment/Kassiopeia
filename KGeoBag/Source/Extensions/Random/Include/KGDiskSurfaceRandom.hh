/*
 * KGDiskSurfaceRandom.hh
 *
 *  Created on: 26.09.2014
 *      Author: J. Behrens
 */

#ifndef KGDISKSURFACERANDOM_HH_
#define KGDISKSURFACERANDOM_HH_

#include "KGDiskSurface.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
     * \brief Class for dicing a point on a KGDiskSurface.
     */
class KGDiskSurfaceRandom : virtual public KGShapeRandom, public KGDiskSurface::Visitor
{
  public:
    KGDiskSurfaceRandom() : KGShapeRandom() {}
    ~KGDiskSurfaceRandom() override {}

    /**
         * \brief Visitor function to dice the point on the KGDiskSurface.
         *
         * \param aDiskSurface
         */
    void VisitDiskSurface(KGDiskSurface* aDiskSurface) override;
};
}  // namespace KGeoBag

#endif /* KGDISKSURFACERANDOM_HH_ */

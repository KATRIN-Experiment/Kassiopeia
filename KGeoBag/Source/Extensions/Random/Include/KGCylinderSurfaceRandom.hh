/*
 * KGCylinderSurfaceRandom.hh
 *
 *  Created on: 20.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGCYLINDERSURFACERANDOM_HH_
#define KGCYLINDERSURFACERANDOM_HH_

#include "KGCylinderSurface.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a point on a KGCylinderSurface.
   */
class KGCylinderSurfaceRandom : virtual public KGShapeRandom, public KGCylinderSurface::Visitor
{
  public:
    KGCylinderSurfaceRandom() : KGShapeRandom() {}
    ~KGCylinderSurfaceRandom() override {}

    /**
     * \brief Visitor function to dice the point on the KGCylinderSpace.
     *
     * \param aCylinderSpace
     */
    void VisitCylinderSurface(KGCylinderSurface* aCylinderSpace) override;
};
}  // namespace KGeoBag

#endif /* KGCYLINDERSURFACERANDOM_HH_ */

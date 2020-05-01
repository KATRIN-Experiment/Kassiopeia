/*
 * KGCylinderSpaceRandom.hh
 *
 *  Created on: 14.05.2014
 *      Author: oertlin
 */

#ifndef KGCYLINDERSPACERANDOM_HH_
#define KGCYLINDERSPACERANDOM_HH_

#include "KGCylinderSpace.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a random point inside
   * the KGCylinderSpace.
   */
class KGCylinderSpaceRandom : virtual public KGShapeRandom, public KGCylinderSpace::Visitor
{
  public:
    KGCylinderSpaceRandom() : KGShapeRandom() {}
    ~KGCylinderSpaceRandom() override {}

    /**
     * \brief Visitor function for dicing the point
     * inside the KGCylinderSpace.
     *
     * \brief aCylinderSpace
     */
    void VisitCylinderSpace(KGCylinderSpace* aCylinderSpace) override;
};
}  // namespace KGeoBag

#endif /* KGCYLINDERSPACERANDOM_HH_ */

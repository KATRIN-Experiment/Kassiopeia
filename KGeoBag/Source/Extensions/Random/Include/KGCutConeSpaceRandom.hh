/*
 * KGCutConeSpaceRandom.hh
 *
 *  Created on: 20.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGCUTCONESPACERANDOM_HH_
#define KGCUTCONESPACERANDOM_HH_

#include "KGCutConeSpace.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a point
   * inside a KGCutConeSpace.
   */
class KGCutConeSpaceRandom : virtual public KGShapeRandom, public KGCutConeSpace::Visitor
{
  public:
    KGCutConeSpaceRandom() : KGShapeRandom() {}
    ~KGCutConeSpaceRandom() override {}

    /**
     * \brief Visitor function for dicing the point
     * inside the KGCutConeSpace.
     *
     * \param aCutConeSpace
     */
    void VisitCutConeSpace(KGCutConeSpace* aCutConeSpace) override;
};
}  // namespace KGeoBag

#endif /* KGCUTCONESPACERANDOM_HH_ */

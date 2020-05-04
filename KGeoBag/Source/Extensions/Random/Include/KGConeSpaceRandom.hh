/*
 * KGConeSpaceRandom.hh
 *
 *  Created on: 20.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGCONESPACERANDOM_HH_
#define KGCONESPACERANDOM_HH_

#include "KGConeSpace.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a point inside a KGConeSpace.
   */
class KGConeSpaceRandom : virtual public KGShapeRandom, public KGConeSpace::Visitor
{
  public:
    KGConeSpaceRandom() : KGShapeRandom() {}
    ~KGConeSpaceRandom() override {}

    /**
     * \brief Visitor function to dice a point
     * insode a KGConeSpace.
     *
     * \param aConeSpace
     */
    void VisitConeSpace(KGConeSpace* aConeSpace) override;
};
}  // namespace KGeoBag

#endif /* KGCONESPACERANDOM_HH_ */

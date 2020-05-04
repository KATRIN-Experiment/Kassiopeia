/*
 * KGGenericSpaceRandom.hh
 *
 *  Created on: 21.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGGENERICSPACERANDOM_HH_
#define KGGENERICSPACERANDOM_HH_

#include "KGRandomMessage.hh"
#include "KGShapeRandom.hh"
#include "KGVolume.hh"

namespace KGeoBag
{
/**
   * \brief Class for implementation of a generic code
   * for dicing a point inside an arbitrary space.
   */
class KGGenericSpaceRandom : virtual public KGShapeRandom, public KGVolume::Visitor
{
  public:
    KGGenericSpaceRandom() : KGShapeRandom() {}
    ~KGGenericSpaceRandom() override {}

    /**
     * \brief Visitor function to dice the point inside
     * an arbitrary space.
     *
     * \brief aVolume
     */
    void VisitVolume(KGVolume* aVolume) override;
};
}  // namespace KGeoBag

#endif /* KGGENERICSPACERANDOM_HH_ */

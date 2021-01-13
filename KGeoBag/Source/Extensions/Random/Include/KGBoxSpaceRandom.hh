/*
 * KGBoxSpaceRandom.hh
 *
 *  Created on: 13.05.2014
 *      Author: oertlin
 */

#ifndef KGBOXSPACERANDOM_HH_
#define KGBOXSPACERANDOM_HH_

#include "KGBoxSpace.hh"
#include "KGShapeRandom.hh"

namespace KGeoBag
{
/**
   * \brief Class for dicing a point inside a KGBoxSpace.
   */
class KGBoxSpaceRandom : virtual public KGShapeRandom, public KGBoxSpace::Visitor
{
  public:
    KGBoxSpaceRandom() : KGShapeRandom() {}
    ~KGBoxSpaceRandom() override = default;

    /**
	 * \brief Visitor function for dicing a point inside
	 * a KGBoxSpace.
	 *
	 * \brief aBox
	 */
    void VisitBoxSpace(const KGBoxSpace* aBox) override;
};
}  // namespace KGeoBag

#endif /* KGBOXSPACERANDOM_HH_ */

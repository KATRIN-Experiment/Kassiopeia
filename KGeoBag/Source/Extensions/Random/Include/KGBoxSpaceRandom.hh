/*
 * KGBoxSpaceRandom.hh
 *
 *  Created on: 13.05.2014
 *      Author: oertlin
 */

#ifndef KGBOXSPACERANDOM_HH_
#define KGBOXSPACERANDOM_HH_

#include "KGShapeRandom.hh"
#include "KGBoxSpace.hh"

namespace KGeoBag
{
  /**
   * \brief Class for dicing a point inside a KGBoxSpace.
   */
  class KGBoxSpaceRandom : virtual public KGShapeRandom,
				 public KGBoxSpace::Visitor
  {
  public:
	  KGBoxSpaceRandom() : KGShapeRandom() {}
    virtual ~KGBoxSpaceRandom() {}

    /**
	 * \brief Visitor function for dicing a point inside
	 * a KGBoxSpace.
	 *
	 * \brief aBox
	 */
    virtual void VisitBoxSpace(const KGBoxSpace* aBox);
  };
}

#endif /* KGBOXSPACERANDOM_HH_ */

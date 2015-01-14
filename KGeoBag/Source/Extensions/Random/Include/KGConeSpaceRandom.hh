/*
 * KGConeSpaceRandom.hh
 *
 *  Created on: 20.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGCONESPACERANDOM_HH_
#define KGCONESPACERANDOM_HH_

#include "KGShapeRandom.hh"
#include "KGConeSpace.hh"

namespace KGeoBag
{
  /**
   * \brief Class for dicing a point inside a KGConeSpace.
   */
  class KGConeSpaceRandom : virtual public KGShapeRandom,
				 public KGConeSpace::Visitor
  {
  public:
	  KGConeSpaceRandom() : KGShapeRandom() {}
    virtual ~KGConeSpaceRandom() {}

    /**
     * \brief Visitor function to dice a point
     * insode a KGConeSpace.
     *
     * \param aConeSpace
     */
    virtual void VisitConeSpace(KGConeSpace* aConeSpace);
  };
}

#endif /* KGCONESPACERANDOM_HH_ */

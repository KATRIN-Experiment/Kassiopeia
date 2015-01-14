/*
 * KGBoxSurfaceRandom.hh
 *
 *  Created on: 12.05.2014
 *      Author: oertlin
 */

#ifndef KGBOXSURFACERANDOM_HH_
#define KGBOXSURFACERANDOM_HH_

#include "KGShapeRandom.hh"
#include "KGBox.hh"

namespace KGeoBag
{
  /**
   * \brief Class for dicing a point on a KGBoxSurface.
   */
  class KGBoxSurfaceRandom : virtual public KGShapeRandom,
				 public KGBox::Visitor
  {
  public:
	  KGBoxSurfaceRandom() : KGShapeRandom() {}
    virtual ~KGBoxSurfaceRandom() {}

    /**
     * \brief Visitor function for dicing a point on
     * a KGBox.
     *
     * \brief aBox
     */
    virtual void VisitBox(KGBox* aBox);
  };
}

#endif /* KGBOXSURFACERANDOM_HH_ */

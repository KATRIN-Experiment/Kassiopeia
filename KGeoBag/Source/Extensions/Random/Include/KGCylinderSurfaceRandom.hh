/*
 * KGCylinderSurfaceRandom.hh
 *
 *  Created on: 20.05.2014
 *      Author: Jan Oertlin
 */

#ifndef KGCYLINDERSURFACERANDOM_HH_
#define KGCYLINDERSURFACERANDOM_HH_

#include "KGShapeRandom.hh"
#include "KGCylinderSurface.hh"

namespace KGeoBag
{
  /**
   * \brief Class for dicing a point on a KGCylinderSurface.
   */
  class KGCylinderSurfaceRandom : virtual public KGShapeRandom,
				 public KGCylinderSurface::Visitor
  {
  public:
	  KGCylinderSurfaceRandom() : KGShapeRandom() {}
    virtual ~KGCylinderSurfaceRandom() {}

    /**
     * \brief Visitor function to dice the point on the KGCylinderSpace.
     *
     * \param aCylinderSpace
     */
    virtual void VisitCylinderSurface(KGCylinderSurface* aCylinderSpace);
  };
}

#endif /* KGCYLINDERSURFACERANDOM_HH_ */

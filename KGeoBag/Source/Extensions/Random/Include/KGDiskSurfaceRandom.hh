/*
 * KGDiskSurfaceRandom.hh
 *
 *  Created on: 26.09.2014
 *      Author: J. Behrens
 */

#ifndef KGDISKSURFACERANDOM_HH_
#define KGDISKSURFACERANDOM_HH_

#include "KGShapeRandom.hh"
#include "KGDiskSurface.hh"

namespace KGeoBag
{
    /**
     * \brief Class for dicing a point on a KGDiskSurface.
     */
    class KGDiskSurfaceRandom : virtual public KGShapeRandom,
    public KGDiskSurface::Visitor
    {
    public:
        KGDiskSurfaceRandom() : KGShapeRandom() {}
        virtual ~KGDiskSurfaceRandom() {}

        /**
         * \brief Visitor function to dice the point on the KGDiskSurface.
         *
         * \param aDiskSurface
         */
        virtual void VisitDiskSurface(KGDiskSurface* aDiskSurface);
    };
}

#endif /* KGDISKSURFACERANDOM_HH_ */

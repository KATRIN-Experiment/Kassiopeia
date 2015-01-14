#ifndef KGPLANARCLOSEDPATH_HH_
#define KGPLANARCLOSEDPATH_HH_

#include "KGPlanarPath.hh"

namespace KGeoBag
{

    class KGPlanarClosedPath :
        public KGPlanarPath
    {
        public:
            KGPlanarClosedPath();
            virtual ~KGPlanarClosedPath();

            //**********
            //properties
            //**********

        public:
            virtual const KTwoVector& Anchor() const = 0;
    };

}

#endif

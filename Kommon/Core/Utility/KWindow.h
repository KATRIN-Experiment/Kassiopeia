#ifndef _katrin_KWindow_h_
#define _katrin_KWindow_h_

#include "KNamed.h"

// hello reader,
// this is a really stupid base class for windows.
// sorry to waste your time.

namespace katrin
{

    class KPainter;

    class KWindow :
        public KNamed
    {
        public:
            KWindow();
            virtual ~KWindow();

        public:
            virtual void Render() = 0;
            virtual void Display() = 0;
            virtual void Write() = 0;

            virtual void AddPainter( KPainter* aPainter ) = 0;
            virtual void RemovePainter( KPainter* aPainter ) = 0;
    };

}

#endif

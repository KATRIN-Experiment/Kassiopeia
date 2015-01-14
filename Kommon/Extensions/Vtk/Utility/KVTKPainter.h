#ifndef _katrin_KVTKPainter_h_
#define _katrin_KVTKPainter_h_

#include "KPainter.h"

namespace katrin
{
    class KVTKWindow;

    class KVTKPainter :
        public KPainter
    {
        public:
            KVTKPainter();
            virtual ~KVTKPainter();

        public:
            void SetWindow( KWindow* aWindow );
            void ClearWindow( KWindow* aWindow );

            void SetDisplayMode( bool aMode );
            void SetWriteMode( bool aMode );

        protected:
            KVTKWindow* fWindow;
            bool fDisplayEnabled;
            bool fWriteEnabled;
    };

}

#endif

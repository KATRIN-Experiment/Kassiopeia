#ifndef _katrin_KROOTPainter_h_
#define _katrin_KROOTPainter_h_

#include "KPainter.h"

namespace katrin
{
    class KROOTWindow;

    class KROOTPainter :
        public KPainter
    {
        public:
    		KROOTPainter();
            virtual ~KROOTPainter();

        public:
            void SetWindow( KWindow* aWindow );
            void ClearWindow( KWindow* aWindow );

            void SetDisplayMode( bool aMode );
            void SetWriteMode( bool aMode );

            virtual double GetXMin() = 0;
            virtual double GetXMax() = 0;
            virtual double GetYMin() = 0;
            virtual double GetYMax() = 0;

            virtual std::string GetXAxisLabel() = 0;
            virtual std::string GetYAxisLabel() = 0;

        protected:
            KROOTWindow* fWindow;
            bool fDisplayEnabled;
            bool fWriteEnabled;
    };

}

#endif

#ifndef _Kassiopeia_KSReadTrackROOT_h_
#define _Kassiopeia_KSReadTrackROOT_h_

#include "KSReadIteratorROOT.h"

namespace Kassiopeia
{

    class KSReadTrackROOT :
        public KSReadIteratorROOT
    {
        public:
            KSReadTrackROOT( TFile* aFile );
            virtual ~KSReadTrackROOT();

        public:
            unsigned int GetTrackIndex() const;
            unsigned int GetFirstStepIndex() const;
            unsigned int GetLastStepIndex() const;

        public:
            KSReadTrackROOT& operator= (const unsigned int& aValue);

        private:
            unsigned int fTrackIndex;
            unsigned int fFirstStepIndex;
            unsigned int fLastStepIndex;
    };

}

#endif

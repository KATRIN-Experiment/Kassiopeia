#ifndef _Kassiopeia_KSReadRunROOT_h_
#define _Kassiopeia_KSReadRunROOT_h_

#include "KSReadIteratorROOT.h"

namespace Kassiopeia
{

    class KSReadRunROOT :
        public KSReadIteratorROOT
    {
        public:
            KSReadRunROOT( TFile* aFile );
            virtual ~KSReadRunROOT();

        public:
            unsigned int GetRunIndex() const;
            unsigned int GetLastRunIndex() const;
            unsigned int GetFirstEventIndex() const;
            unsigned int GetLastEventIndex() const;
            unsigned int GetFirstTrackIndex() const;
            unsigned int GetLastTrackIndex() const;
            unsigned int GetFirstStepIndex() const;
            unsigned int GetLastStepIndex() const;

        public:
            KSReadRunROOT& operator= (const unsigned int& aValue);

        private:
            unsigned int fRunIndex;
            unsigned int fFirstEventIndex;
            unsigned int fLastEventIndex;
            unsigned int fFirstTrackIndex;
            unsigned int fLastTrackIndex;
            unsigned int fFirstStepIndex;
            unsigned int fLastStepIndex;
    };

}

#endif

#ifndef Kassiopeia_KSTrackModifier_h_
#define Kassiopeia_KSTrackModifier_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{
    class KSTrack;

    class KSTrackModifier:
            public KSComponentTemplate< KSTrackModifier >
    {
        public:
            KSTrackModifier();
            virtual ~KSTrackModifier();

        public:

            //returns true if any of the state variables of aTrack are changed
            virtual bool ExecutePreTrackModification( KSTrack& aTrack ) = 0;

            //returns true if any of the state variables of aTrack are changed
            virtual bool ExecutePostTrackModification( KSTrack& aTrack ) = 0;
    };

}

#endif

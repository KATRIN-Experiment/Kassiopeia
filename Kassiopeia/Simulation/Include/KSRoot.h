#ifndef Kassiopeia_KSRoot_h_
#define Kassiopeia_KSRoot_h_

#include "KSComponentTemplate.h"

#include "KSMainMessage.h"

namespace Kassiopeia
{

    class KSToolbox;
    class KSRootMagneticField;
    class KSRootElectricField;
    class KSRootSpace;
    class KSRootGenerator;
    class KSRootTrajectory;
    class KSRootSpaceInteraction;
    class KSRootSpaceNavigator;
    class KSRootSurfaceInteraction;
    class KSRootSurfaceNavigator;
    class KSRootTerminator;
    class KSRootWriter;
    class KSRootStepModifier;

    class KSSimulation;
    class KSRun;
    class KSEvent;
    class KSTrack;
    class KSStep;

    class KSRoot :
        public KSComponentTemplate< KSRoot >
    {
        public:
            KSRoot();
            KSRoot( const KSRoot& aCopy );
            KSRoot* Clone() const;
            virtual ~KSRoot();

        public:
            void Execute( KSSimulation* aSimulation );

        protected:
            void ExecuteRun();
            void ExecuteEvent();
            void ExecuteTrack();
            void ExecuteStep();

        protected:
            void ActivateComponent();
            void DeactivateComponent();
            void InitializeComponent();
            void DeinitializeComponent();

        private:
            KSSimulation* fSimulation;
            KSRun* fRun;
            KSEvent* fEvent;
            KSTrack* fTrack;
            KSStep* fStep;

            KSToolbox* fToolbox;
            KSRootMagneticField* fRootMagneticField;
            KSRootElectricField* fRootElectricField;
            KSRootSpace* fRootSpace;
            KSRootGenerator* fRootGenerator;
            KSRootTrajectory* fRootTrajectory;
            KSRootSpaceInteraction* fRootSpaceInteraction;
            KSRootSpaceNavigator* fRootSpaceNavigator;
            KSRootSurfaceInteraction* fRootSurfaceInteraction;
            KSRootSurfaceNavigator* fRootSurfaceNavigator;
            KSRootTerminator* fRootTerminator;
            KSRootWriter* fRootWriter;
            KSRootStepModifier* fRootStepModifier;

            unsigned int fRunIndex;
            unsigned int fEventIndex;
            unsigned int fTrackIndex;
            unsigned int fStepIndex;
    };

}

#endif

#ifndef Kassiopeia_KSRoot_h_
#define Kassiopeia_KSRoot_h_

#include "KSComponentTemplate.h"

#include "KSMainMessage.h"

#include "gsl/gsl_errno.h"
#include "KToolbox.h"

namespace Kassiopeia
{
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
    class KSRootTrackModifier;
    class KSRootEventModifier;
    class KSRootRunModifier;

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
            static void SignalHandler(int aSignal);
            static void GSLErrorHandler(const char* aReason, const char* aFile, int aLine, int aErrNo);

        private:
            katrin::KToolbox& fToolbox;

            KSSimulation* fSimulation;
            KSRun* fRun;
            KSEvent* fEvent;
            KSTrack* fTrack;
            KSStep* fStep;

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
            KSRootTrackModifier* fRootTrackModifier;
            KSRootEventModifier* fRootEventModifier;
            KSRootRunModifier* fRootRunModifier;

            bool fOnce;
            bool fRestartNavigation;

            unsigned int fRunIndex;
            unsigned int fEventIndex;
            unsigned int fTrackIndex;
            unsigned int fStepIndex;

            static bool fStopRunSignal;
            static bool fStopEventSignal;
            static bool fStopTrackSignal;
            static std::string fStopSignalName;

            gsl_error_handler_t* fDefaultGSLErrorHandler;
            static bool fGSLErrorSignal;
            static std::string fGSLErrorString;
    };

}

#endif

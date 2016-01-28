#ifndef Kassiopeia_KSStep_h_
#define Kassiopeia_KSStep_h_

#include "KSComponentTemplate.h"
#include "KSParticle.h"
#include "KField.h"

namespace Kassiopeia
{

    class KSTrack;

    class KSStep :
        public KSComponentTemplate< KSStep >
    {
        public:
            KSStep();
            KSStep( const KSStep& aCopy );
            KSStep& operator=( const KSStep& aCopy );
            KSStep* Clone() const;
            ~KSStep();

            void Reset();

            //***
            //IDs
            //***

        public:
            K_REFS( unsigned int, StepId )
            K_REFS( unsigned int, StepCount )
            K_REFS( unsigned int, ParentTrackId )

            //****
            //step
            //****

        public:
            K_REFS( double, ContinuousTime )
            K_REFS( double, ContinuousLength )
            K_REFS( double, ContinuousEnergyChange )
            K_REFS( double, ContinuousMomentumChange )
            K_REFS( unsigned int, DiscreteSecondaries )
            K_REFS( double, DiscreteEnergyChange )
            K_REFS( double, DiscreteMomentumChange )

            //**********
            //terminator
            //**********

        public:
            K_REFS( string, TerminatorName )
            K_REFS( bool, TerminatorFlag )

            //**********
            //trajectory
            //**********

        public:
            K_REFS( string, TrajectoryName )
            K_REFS( KThreeVector, TrajectoryCenter )
            K_REFS( double, TrajectoryRadius )
            K_REFS( double, TrajectoryStep )

            //*****************
            //space interaction
            //*****************

        public:
            K_REFS( string, SpaceInteractionName )
            K_REFS( double, SpaceInteractionStep )
            K_REFS( bool, SpaceInteractionFlag )

            //****************
            //space navigation
            //****************

        public:
            K_REFS( string, SpaceNavigationName )
            K_REFS( double, SpaceNavigationStep )
            K_REFS( bool, SpaceNavigationFlag )

            //*******************
            //surface interaction
            //*******************

        public:
            K_REFS( string, SurfaceInteractionName )
            K_REFS( bool, SurfaceInteractionFlag )

            //******************
            //surface navigation
            //******************

        public:
            K_REFS( string, SurfaceNavigationName )
            K_REFS( bool, SurfaceNavigationFlag )

            //*********
            //particles
            //*********

        public:
            K_REFS( KSParticle, InitialParticle )
            K_REFS( KSParticle, TerminatorParticle )
            K_REFS( KSParticle, TrajectoryParticle )
            K_REFS( KSParticle, InteractionParticle )
            K_REFS( KSParticle, NavigationParticle )
            K_REFS( KSParticle, FinalParticle )

            //*****
            //queue
            //*****

        public:
            K_REFS( KSParticleQueue, ParticleQueue )

    };

}

#endif

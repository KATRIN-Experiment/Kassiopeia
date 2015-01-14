#ifndef Kassiopeia_KSRootTerminator_h_
#define Kassiopeia_KSRootTerminator_h_

#include "KSTerminator.h"
#include "KSStep.h"
#include "KSList.h"

namespace Kassiopeia
{

    class KSTrack;

    class KSRootTerminator :
        public KSComponentTemplate< KSRootTerminator, KSTerminator >
    {
        public:
            KSRootTerminator();
            KSRootTerminator( const KSRootTerminator& aCopy );
            KSRootTerminator* Clone() const;
            virtual ~KSRootTerminator();

            //**********
            //terminator
            //**********

        public:
            void CalculateTermination( const KSParticle& anInitialParticle, bool& aFlag );
            void ExecuteTermination( const KSParticle& anInitialParticle, KSParticle& aFinalParticle, KSParticleQueue& aQueue ) const;

            //***********
            //composition
            //***********

        public:
            void AddTerminator( KSTerminator* aTerminator );
            void RemoveTerminator( KSTerminator* aTerminator );

        private:
            KSList< KSTerminator > fTerminators;
            KSTerminator* fTerminator;

            //******
            //action
            //******

        public:
            void SetStep( KSStep* aStep );

            void CalculateTermination();
            void ExecuteTermination();

            virtual void PushUpdateComponent();
            virtual void PushDeupdateComponent();

        private:
            KSStep* fStep;
            const KSParticle* fInitialParticle;
            KSParticle* fTerminatorParticle;
            KSParticle* fFinalParticle;
            KSParticleQueue* fParticleQueue;
    };


}

#endif

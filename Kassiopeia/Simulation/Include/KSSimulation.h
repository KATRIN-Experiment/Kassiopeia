#ifndef Kassiopeia_KSSimulation_h_
#define Kassiopeia_KSSimulation_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{

    class KSSimulation :
        public KSComponentTemplate< KSSimulation >
    {
        public:
            KSSimulation();
            KSSimulation( const KSSimulation& aCopy );
            KSSimulation* Clone() const;
            virtual ~KSSimulation();

        public:
            void SetSeed( const unsigned int& aSeed );
            const unsigned int& GetSeed() const;

            void SetRun( const unsigned int& aRun );
            const unsigned int& GetRun() const;

            void SetEvents( const unsigned int& anEvents );
            const unsigned int& GetEvents() const;

            void AddCommand( KSCommand* aCommand );
            void RemoveCommand( KSCommand* aCommand );

        protected:
            void InitializeComponent();
            void DeinitializeComponent();
            void ActivateComponent();
            void DeactivateComponent();

            unsigned int fSeed;
            unsigned int fRun;
            unsigned int fEvents;
            vector< KSCommand* > fCommands;
    };

}

#endif

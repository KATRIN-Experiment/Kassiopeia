#ifndef Kassiopeia_KSGenGeneratorComposite_h_
#define Kassiopeia_KSGenGeneratorComposite_h_

#include "KSGenerator.h"
#include "KSGenValue.h"
#include "KSGenCreator.h"
#include "KSList.h"

#include "KSGenSpecial.h"

namespace Kassiopeia
{

    class KSGenGeneratorComposite :
        public KSComponentTemplate< KSGenGeneratorComposite, KSGenerator >
    {
        public:
            KSGenGeneratorComposite();
            KSGenGeneratorComposite( const KSGenGeneratorComposite& aCopy );
            KSGenGeneratorComposite* Clone() const;
            virtual ~KSGenGeneratorComposite();

            //******
            //action
            //******

        public:
            virtual void ExecuteGeneration( KSParticleQueue& aPrimaries );

            //***********
            //composition
            //***********

        public:            
            void SetPid( KSGenValue* aPidValue );
            KSGenValue* GetPid();

            void AddCreator( KSGenCreator* aCreator );
            void RemoveCreator( KSGenCreator* aCreator );

            void AddSpecial( KSGenSpecial* a );
            void RemoveSpecial( KSGenSpecial* a );

        private:
            void InitializeComponent();
            void DeinitializeComponent();

        protected:
            KSGenValue* fPidValue;
            KSList< KSGenSpecial > fSpecials;
            KSList< KSGenCreator > fCreators;
    };

}

#endif

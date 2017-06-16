#ifndef Kassiopeia_KSComponentGroup_h_
#define Kassiopeia_KSComponentGroup_h_

#include "KSComponent.h"

namespace Kassiopeia
{

    class KSComponentGroup :
        public KSComponent
    {
        public:
            KSComponentGroup();
            KSComponentGroup( const KSComponentGroup& aCopy );
            virtual ~KSComponentGroup();

        public:
            KSComponentGroup* Clone() const;
            KSComponent* Component( const std::string& aField );
            KSCommand* Command( const std::string& aField, KSComponent* aChild );

        public:
            void AddComponent( KSComponent* aComponent );
            void RemoveComponent( KSComponent* aComponent );

            KSComponent* ComponentAt( unsigned int anIndex );
            const KSComponent* ComponentAt( unsigned int anIndex ) const;
            unsigned int ComponentCount() const;

        private:
            typedef std::vector< KSComponent* > ComponentVector;
            typedef ComponentVector::iterator ComponentIt;
            typedef ComponentVector::const_iterator ComponentCIt;

            ComponentVector fComponents;
    };

}

#endif

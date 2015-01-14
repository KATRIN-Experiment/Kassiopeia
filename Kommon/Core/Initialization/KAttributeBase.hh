#ifndef Kommon_KAttributeBase_hh_
#define Kommon_KAttributeBase_hh_

#include "KContainer.hh"
#include "KProcessor.hh"

namespace katrin
{

    class KElementBase;

    class KAttributeBase :
        public KContainer,
        public KProcessor
    {
        public:
            KAttributeBase();
            virtual ~KAttributeBase();

        public:
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KErrorToken* aToken );

        public:
            virtual bool SetValue( KToken* aValue ) = 0;

        protected:
            KElementBase* fParentElement;
    };
}



#endif

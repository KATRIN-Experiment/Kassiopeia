#ifndef Kommon_KElementBase_hh_
#define Kommon_KElementBase_hh_

#include "KContainer.hh"
#include "KProcessor.hh"

#include <map>

namespace katrin
{
    class KAttributeBase;
    class KElementBase;

    typedef std::map< std::string, KAttributeBase* (*)( KElementBase* ) > KAttributeMap;
    typedef KAttributeMap::value_type KAttributeEntry;
    typedef KAttributeMap::iterator KAttributeIt;
    typedef KAttributeMap::const_iterator KAttributeCIt;

    typedef std::map< std::string, KElementBase* (*)( KElementBase* ) > KElementMap;
    typedef KElementMap::value_type KElementEntry;
    typedef KElementMap::iterator KElementIt;
    typedef KElementMap::const_iterator KElementCIt;

    class KElementBase :
        public KContainer,
        public KProcessor
    {
        public:
            KElementBase();
            virtual ~KElementBase();

        public:
            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );
            virtual void ProcessToken( KErrorToken* aToken );

        public:
            virtual bool Begin() = 0;
            virtual bool AddAttribute( KContainer* aToken ) = 0;
            virtual bool Body() = 0;
            virtual bool AddElement( KContainer* anElement ) = 0;
            virtual bool End() = 0;
            virtual bool SetValue( KToken* aValue ) = 0;

        protected:
            KElementBase* fParentElement;

            const KAttributeMap* fAttributes;
            KAttributeBase* fChildAttribute;
            unsigned int fAttributeDepth;

            const KElementMap* fElements;
            KElementBase* fChildElement;
            unsigned int fElementDepth;
    };
}

#endif

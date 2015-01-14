#ifndef Kommon_KTypedTokens_hh_
#define Kommon_KTypedTokens_hh_

#include "KToken.hh"

#include <cstdlib>

namespace katrin
{

    template< class XType >
    class KTypedToken :
        public KToken
    {
        public:
            KTypedToken();
            KTypedToken( const KTypedToken& aToken );
            virtual ~KTypedToken();

            virtual KToken* Clone();
    };

    template< class XType >
    inline KTypedToken< XType >::KTypedToken() :
            KToken()
    {
    }
    template< class XType >
    inline KTypedToken< XType >::KTypedToken( const KTypedToken& aToken ) :
            KToken( aToken )
    {
    }
    template< class XType >
    KTypedToken< XType >::~KTypedToken()
    {
    }

    template< class XType >
    inline KToken* KTypedToken< XType >::Clone()
    {
        return new KTypedToken< XType >( *this );
    }

    class KBeginParsing;
    typedef KTypedToken< KBeginParsing > KBeginParsingToken;

    class KEndParsing;
    typedef KTypedToken< KEndParsing > KEndParsingToken;

    class KBeginFile;
    typedef KTypedToken< KBeginFile > KBeginFileToken;

    class KEndFile;
    typedef KTypedToken< KEndFile > KEndFileToken;

    class KBeginElement;
    typedef KTypedToken< KBeginElement > KBeginElementToken;

    class KBeginAttribute;
    typedef KTypedToken< KBeginAttribute > KBeginAttributeToken;

    class KAttributeData;
    typedef KTypedToken< KAttributeData > KAttributeDataToken;

    class KEndAttribute;
    typedef KTypedToken< KEndAttribute > KEndAttributeToken;

    class KMidElement;
    typedef KTypedToken< KMidElement > KMidElementToken;

    class KElementData;
    typedef KTypedToken< KElementData > KElementDataToken;

    class KEndElement;
    typedef KTypedToken< KEndElement > KEndElementToken;

    class KComment;
    typedef KTypedToken< KComment > KCommentToken;

    class KError;
    typedef KTypedToken< KError > KErrorToken;

}

#endif

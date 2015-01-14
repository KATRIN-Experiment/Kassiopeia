#ifndef Kommon_KChattyProcessor_hh_
#define Kommon_KChattyProcessor_hh_

#include "KProcessor.hh"

#include <cstdlib>

namespace katrin
{
    class KChattyProcessor :
        public KProcessor
    {
        public:
            KChattyProcessor();
            virtual ~KChattyProcessor();

            virtual void ProcessToken( KBeginParsingToken* aToken );
            virtual void ProcessToken( KBeginFileToken* aToken );
            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );
            virtual void ProcessToken( KEndFileToken* aToken );
            virtual void ProcessToken( KEndParsingToken* aToken );
            virtual void ProcessToken( KCommentToken* aToken );
            virtual void ProcessToken( KErrorToken* aToken );

    };
}

#endif

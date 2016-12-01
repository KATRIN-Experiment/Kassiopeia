#ifndef Kommon_KTagProcessor_hh_
#define Kommon_KTagProcessor_hh_

#include "KProcessor.hh"

#include <stack>
#include <vector>

namespace katrin
{

    class KTagProcessor :
        public KProcessor
    {
        public:
            KTagProcessor();
            virtual ~KTagProcessor();

            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );

        private:
            typedef enum
            {
                eInactive, eActive
            } State;
            State fState;

            std::vector< std::string > fTags;
            std::vector< std::string >::iterator fTagIt;
            std::stack< std::vector< std::string > > fTagStack;
    };

}

#endif

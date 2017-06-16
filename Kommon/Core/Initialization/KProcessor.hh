#ifndef Kommon_KProcessor_hh_
#define Kommon_KProcessor_hh_

#include "KTypedTokens.hh"

namespace katrin
{

    class KProcessor
    {
        public:
            KProcessor();
            virtual ~KProcessor();

            //*****************
            //structural system
            //*****************

        public:
            static void Connect( KProcessor* aParent, KProcessor* aChild );
            static void Disconnect( KProcessor* aParent, KProcessor* aChild );

            void InsertBefore( KProcessor* aTarget );
            void InsertAfter( KProcessor* aTarget );
            void Remove();

            KProcessor* GetFirstParent();
            KProcessor* GetParent();

            KProcessor* GetLastChild();
            KProcessor* GetChild();

        protected:
            KProcessor* fParent;
            KProcessor* fChild;

            //*****************
            //processing system
            //*****************

        public:
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

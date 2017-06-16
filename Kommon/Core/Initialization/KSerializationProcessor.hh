#ifndef Kommon_KSerializationProcessor_hh_
#define Kommon_KSerializationProcessor_hh_

#include "KProcessor.hh"

namespace katrin
{
    class KSerializationProcessor :
        public KProcessor

    {
        public:
    		KSerializationProcessor();
            virtual ~KSerializationProcessor();

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

            std::string GetConfig() { return completeconfig; }
            void Clear() { completeconfig.clear(); }

       private:
            std::string completeconfig;
            std::string fOutputFilename;

            typedef enum
            {
                eElementInactive, eActiveFileDefine, eElementComplete
            } ElementState;
            typedef enum
            {
                eAttributeInactive, eActiveFileName, eAttributeComplete
            } AttributeState;

            ElementState fElementState;
            AttributeState fAttributeState;

    };
}

#endif

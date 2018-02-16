#ifndef Kommon_KIncludeProcessor_hh_
#define Kommon_KIncludeProcessor_hh_

#include "KProcessor.hh"

#include <vector>

namespace katrin
{

    class KIncludeProcessor :
        public KProcessor
    {
        public:
            KIncludeProcessor();
            virtual ~KIncludeProcessor();

            virtual void ProcessToken( KBeginElementToken* aToken );
            virtual void ProcessToken( KBeginAttributeToken* aToken );
            virtual void ProcessToken( KAttributeDataToken* aToken );
            virtual void ProcessToken( KEndAttributeToken* aToken );
            virtual void ProcessToken( KMidElementToken* aToken );
            virtual void ProcessToken( KElementDataToken* aToken );
            virtual void ProcessToken( KEndElementToken* aToken );

            void AddDefaultPath(const std::string& path);

        private:
            void Reset();

            typedef enum
            {
                eElementInactive, eActive, eElementComplete
            } ElementState;
            ElementState fElementState;

            typedef enum
            {
                eAttributeInactive, eName, ePath, eBase, eAttributeComplete
            } AttributeState;
            AttributeState fAttributeState;

            std::vector< std::string > fNames;
            std::vector< std::string > fPaths;
            std::vector< std::string > fBases;

            std::vector< std::string > fDefaultPaths;

            std::vector< std::string > fIncludedPaths;
    };

}

#endif

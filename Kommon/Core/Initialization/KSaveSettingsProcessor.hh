#ifndef Kommon_KSaveSettingsProcessor_hh_
#define Kommon_KSaveSettingsProcessor_hh_

#include "KProcessor.hh"
#include "KToken.hh"

#include <cstdlib>
#include <string>
#include <vector>

namespace katrin
{
    class KSaveSettingsProcessor :
        public KProcessor
    {
        public:
            KSaveSettingsProcessor();
            virtual ~KSaveSettingsProcessor();

            string EscapeToken(string aToken, bool isDirectory=false);

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

			static const std::vector<string>& getCommands()
			{
				return commands;
			}
			static const std::vector<string>& getValues()
			{
				return values;
			}

        private:
            bool fisElement;
            string fXML_value;
            string flastType;
            static std::vector<string> commands;
            static std::vector<string> values;

    };
}

#endif

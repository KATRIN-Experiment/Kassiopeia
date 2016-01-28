#include "KSaveSettingsProcessor.hh"

std::vector<string> katrin::KSaveSettingsProcessor::commands;
std::vector<string> katrin::KSaveSettingsProcessor::values;

namespace katrin
{
	KSaveSettingsProcessor::KSaveSettingsProcessor()
    {
		fisElement = false;
    }

    KSaveSettingsProcessor::~KSaveSettingsProcessor()
    {
    }

    string KSaveSettingsProcessor::EscapeToken(string aToken, bool isDirectory)
	{
		string escapedToken = aToken;
	    size_t pos = 0;

		if(escapedToken != "")
		{
			size_t p = escapedToken.find_first_not_of(" \t");
			escapedToken.erase(0, p);
			p = escapedToken.find_last_not_of(" \t");
			if (string::npos != p)
				escapedToken.erase(p + 1);

			if (isDirectory)
			{
				while ((pos = escapedToken.find("/", pos)) != std::string::npos) {
					escapedToken.replace(pos, 1, "\\");
					pos += 1;
				}
			}

		}
		return(escapedToken);
	}

    void KSaveSettingsProcessor::ProcessToken( KBeginParsingToken* aToken)
    {
		flastType="KBeginParsing";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KBeginFileToken* aToken )
    {
    	flastType="KBeginFile";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KBeginElementToken* aToken )
    {
    	if(EscapeToken(fXML_value) != "")
    	{
    		if(flastType=="KMidElement")
    		{
    			commands.push_back("mkdir");values.push_back(EscapeToken(fXML_value, true));
    			commands.push_back("cd");values.push_back(EscapeToken(fXML_value, true));
    		}
    	}
    	fXML_value = aToken->GetValue();

    	flastType="KBeginElement";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KBeginAttributeToken* aToken )
    {
    	fXML_value = fXML_value + " " + aToken->GetValue() + "=\"";
    	flastType="KBeginAttribute";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KAttributeDataToken* aToken )
    {
    	fXML_value = fXML_value + aToken->GetValue();
    	flastType="KAttributeData";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KEndAttributeToken* aToken )
    {
    	fXML_value = fXML_value + "\"";
    	flastType="KEndAttribute";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KMidElementToken* aToken )
    {
    	if (flastType!="KEndElement") flastType="KMidElement";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KElementDataToken* aToken )
    {
    	if(EscapeToken(fXML_value) != "")
    	{
    		if(flastType=="KMidElement")
    		{
    			commands.push_back("mkdir");values.push_back(EscapeToken(fXML_value, true));
    			commands.push_back("cd");values.push_back(EscapeToken(fXML_value, true));
    			fXML_value="";
    		}
    	}
    	if(EscapeToken(aToken->GetValue()) != "")
		{
    		commands.push_back("tobjstring");values.push_back(EscapeToken(aToken->GetValue()));
		}
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KEndElementToken* aToken )
    {
    	if (flastType=="KEndElement")
		{
        	if(EscapeToken(fXML_value) != "")
        	{
        		commands.push_back("tobjstring");values.push_back(EscapeToken(fXML_value));
        		fXML_value="";
        	}
        	commands.push_back("cd");values.push_back("..");
		}
    	if (flastType=="KMidElement")
    	{
   			commands.push_back("tobjstring");values.push_back(EscapeToken(fXML_value));
   			fXML_value="";
    	}
    	flastType="KEndElement";
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KEndFileToken* aToken )
    {
    	KProcessor::ProcessToken( aToken );
        return;
    }

    void KSaveSettingsProcessor::ProcessToken( KEndParsingToken* aToken)
    {
        KProcessor::ProcessToken( aToken );
        return;
    }
    void KSaveSettingsProcessor::ProcessToken( KCommentToken* aToken )
    {
        KProcessor::ProcessToken( aToken );
        return;
    }

    void KSaveSettingsProcessor::ProcessToken( KErrorToken* aToken )
    {
        KProcessor::ProcessToken( aToken );
        return;
    }

}

#ifndef Kommon_KCommandLineTokenizer_hh_
#define Kommon_KCommandLineTokenizer_hh_

#include <string>
#include <vector>
#include <map>

namespace katrin
{

    class KCommandLineTokenizer
    {
        public:
            KCommandLineTokenizer();
            virtual ~KCommandLineTokenizer();

            //**********
            //processing
            //**********

        public:
            void ProcessCommandLine( int anArgc = 0, char** anArgv = nullptr );

            const std::vector< std::string >& GetFiles();
            const std::map< std::string, std::string >& GetVariables();

        private:
            std::vector< std::string > fFiles;
            std::map< std::string, std::string > fVariables;
    };

    inline const std::vector< std::string >& KCommandLineTokenizer::GetFiles()
    {
        return fFiles;
    }
    inline const std::map< std::string, std::string >& KCommandLineTokenizer::GetVariables()
    {
        return fVariables;
    }

}

#endif

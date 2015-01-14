#ifndef Kommon_KCommandLineTokenizer_hh_
#define Kommon_KCommandLineTokenizer_hh_

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <map>
using std::map;
using std::pair;

#include <cstdlib>

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
            void ProcessCommandLine( int anArgc, char** anArgv );

            const vector< string >& GetFiles();
            const map< string, string >& GetVariables();

        private:
            vector< string > fFiles;
            map< string, string > fVariables;
    };

    inline const vector< string >& KCommandLineTokenizer::GetFiles()
    {
        return fFiles;
    }
    inline const map< string, string >& KCommandLineTokenizer::GetVariables()
    {
        return fVariables;
    }

}

#endif

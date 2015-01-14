#ifndef Kommon_KFormulaProcessor_hh_
#define Kommon_KFormulaProcessor_hh_

#include "KProcessor.hh"

#include <stack>
using std::stack;

#include <map>
using std::map;

namespace katrin
{

    class KFormulaProcessor :
         public KProcessor
     {
         public:
             KFormulaProcessor();
             virtual ~KFormulaProcessor();

             virtual void ProcessToken( KAttributeDataToken* aToken );
             virtual void ProcessToken( KElementDataToken* aToken );

         private:
             void Evaluate( KToken* aToken );

             static const string fStartBracket;
             static const string fEndBracket;
             static const string fEqual;
             static const string fNonEqual;
             static const string fGreater;
             static const string fLess;
             static const string fGreaterEqual;
             static const string fLessEqual;
             static const string fModulo;

     };

}

#endif

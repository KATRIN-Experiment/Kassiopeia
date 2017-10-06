#ifndef Kassiopeia_KSIntCalculatorIonBuilder_h_
#define Kassiopeia_KSIntCalculatorIonBuilder_h_

#include "KComplexElement.hh"
#include "KSIntCalculatorIon.h"
#include "KSInteractionsMessage.h"

using namespace Kassiopeia;
namespace katrin
{ 
    typedef KComplexElement< KSIntCalculatorIon > KSIntCalculatorIonBuilder;

    template< >
    inline bool KSIntCalculatorIonBuilder::AddAttribute( KContainer* aContainer )
    {
      
      if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }

     if( aContainer->GetName() == "gas" )
        {
            aContainer->CopyTo( fObject, &KSIntCalculatorIon::SetGas );
	    
	    if (fObject->GetGas().compare("H_2") != 0) {
	      intmsg(eError) << "\"" << fObject->GetGas() << "\" is not available for ion scattering! Available gases: \"H_2\"" << eom;
	      
	      return false;
	    }
	    
	    else {
	      
	      intmsg(eWarning) << "H_2 ionization only available for the following ions and energies:" << eom;
	      intmsg(eWarning) << "     H^+, D^+, T^+: 75 eV < E < 100 keV" << eom;
	      intmsg(eWarning) << "     H_2^+, D_2^+, T_2^+: 31.6 eV < E < 100 keV" << eom;
	      intmsg(eWarning) << "     H_3^+, D_3^+, T_3^+: 75 eV < E < 100 keV" << eom;
	      intmsg(eWarning) << "     H^-, D^-, T^-: 2.37 eV < E < 50 keV" << eom;
	      
            return true;
	  }
	}

	return false;
    }

}

#endif

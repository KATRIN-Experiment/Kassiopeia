#ifndef KSROOTPOTENTIALPAINTERBUILDER_H
#define KSROOTPOTENTIALPAINTERBUILDER_H

#include "KComplexElement.hh"
#include "KSROOTPotentialPainter.h"

using namespace Kassiopeia;
namespace katrin
{
    typedef KComplexElement< KSROOTPotentialPainter > KSROOTPotentialPainterBuilder;

    template< >
    inline bool KSROOTPotentialPainterBuilder::AddAttribute( KContainer* aContainer )
    {
        if( aContainer->GetName() == "name" )
        {
            aContainer->CopyTo( fObject, &KNamed::SetName );
            return true;
        }
        if( aContainer->GetName() == "x_axis" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetXAxis );
            return true;
        }
        if( aContainer->GetName() == "y_axis" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetYAxis );
            return true;
        }
        if( aContainer->GetName() == "electric_field" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetElectricFieldName );
            return true;
        }
        if( aContainer->GetName() == "r_max" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetRmax );
            return true;
        }
        if( aContainer->GetName() == "z_min" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetZmin );
            return true;
        }
        if( aContainer->GetName() == "z_max" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetZmax );
            return true;
        }
        if( aContainer->GetName() == "r_steps" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetRsteps );
            return true;
        }
        if( aContainer->GetName() == "z_steps" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetZsteps );
            return true;
        }
        if( aContainer->GetName() == "calc_pot" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetCalcPot );
            return true;
        }
        if( aContainer->GetName() == "compare_fields" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetComparison );
            return true;
        }
        if( aContainer->GetName() == "reference_field" )
        {
            aContainer->CopyTo( fObject, &KSROOTPotentialPainter::SetReferenceFieldName );
            return true;
        }
        return false;
    }

}

#endif // KSROOTPOTENTIALPAINTERBUILDER_H

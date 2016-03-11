#include "KSROOTPotentialPainter.h"

#include "KSObject.h"
#include "KSVisualizationMessage.h"
#include "KSToolbox.h"
#include <fstream>
#include <iostream>
#include <limits>

#include "KSElectricField.h"

namespace Kassiopeia
{
    KSROOTPotentialPainter::KSROOTPotentialPainter() :
            fXAxis( "z" ),
            fYAxis( "y" ),
            fCalcPot(1),
            fMap()
    {
    }
    KSROOTPotentialPainter::~KSROOTPotentialPainter()
    {
    }

    void KSROOTPotentialPainter::Render()
    {
        vismsg(eNormal) << "Getting electric field " << fElectricFieldName << " from the toolbox" << eom;
        KSElectricField* tElField = KSToolbox::GetInstance()->GetObjectAs<KSElectricField>( fElectricFieldName );
        if ( tElField == NULL)
            vismsg(eError) << "No electric Field!" << eom;
        vismsg(eNormal) << "Initialize electric field (again)" << eom;
        tElField->Initialize();

        double tDeltaZ = fabs(fZmax-fZmin)/fZsteps;
        double tDeltaR = fabs(fRmax)/fRsteps;
        double tZ,tR;
        TH2D* Map = new TH2D("Map", "Map", fZsteps,fZmin,fZmax,2*fRsteps,-fRmax,fRmax);
        KThreeVector tPosition;

        KThreeVector ElectricField;
        Double_t tPotential;

        vismsg(eNormal) << "start calculating potential map" << eom;
        for( int i=0; i<fZsteps; i++)
        {
            tZ = fZmin + i* tDeltaZ;
            vismsg( eNormal ) << "Electric Field: Z Position: " << i <<"/" << fZsteps << reom;

            for ( int j=fRsteps; j>=0;j--)
            {
                tR = j * tDeltaR;

                if(fYAxis=="y") tPosition.SetComponents(0.,tR,tZ);
                else if(fYAxis=="x") tPosition.SetComponents(tR,0.,tZ);
                else vismsg(eError) << "Please use x or y for the Y-Axis and z for the X-Axis. All other combinations are not yet included" << eom;
                if(fCalcPot==0)
                {
                	tElField->CalculateField(tPosition,0.0,ElectricField);
                    Map->SetBinContent(i+1,fRsteps-j+1,ElectricField.Magnitude());
                    Map->SetBinContent(i+1,fRsteps+j+1,ElectricField.Magnitude());
                }
                else
                {
                	tElField->CalculatePotential(tPosition,0.0,tPotential);
                	Map->SetBinContent(i+1,fRsteps-j+1,tPotential);
                    Map->SetBinContent(i+1,fRsteps+j+1,tPotential);
                }
            }

        }

        fMap=Map;

        return;
    }

    void KSROOTPotentialPainter::Display()
    {
        if( fDisplayEnabled == true )
        {
			fWindow->GetPad()->SetRightMargin(0.15);
        	if(fCalcPot==1) fMap->SetZTitle("Potential (V)");
        	else fMap->SetZTitle("Electric Field (V/m)");
        	if(fXAxis=="z") fMap->GetXaxis()->SetTitle("z (m)");
        	else if(fXAxis=="y") fMap->GetXaxis()->SetTitle("y (m)");
        	else if(fXAxis=="x") fMap->GetXaxis()->SetTitle("x (m)");
        	if(fYAxis=="z") fMap->GetYaxis()->SetTitle("z (m)");
        	else if(fYAxis=="y") fMap->GetYaxis()->SetTitle("y (m)");
        	else if(fYAxis=="x") fMap->GetYaxis()->SetTitle("x (m)");

        	fMap->GetZaxis()->SetTitleOffset(1.4);
        	fMap->SetStats(0);
        	fMap->SetTitle( "" );
        	fMap->Draw( "COLZL" );
        }
        return;
    }

    void KSROOTPotentialPainter::Write()
    {                

    }

    double KSROOTPotentialPainter::GetXMin()
    {
        double tMin( std::numeric_limits< double >::max() );
        return tMin;
    }
    double KSROOTPotentialPainter::GetXMax()
    {
        double tMax( -1.0 * std::numeric_limits< double >::max() );
        return tMax;
    }

    double KSROOTPotentialPainter::GetYMin()
    {
        double tMin( std::numeric_limits< double >::max() );
        return tMin;
    }
    double KSROOTPotentialPainter::GetYMax()
    {
        double tMax( -1.0 * std::numeric_limits< double >::max() );
        return tMax;
    }

    std::string KSROOTPotentialPainter::GetXAxisLabel()
    {
        return fXAxis;
    }

    std::string KSROOTPotentialPainter::GetYAxisLabel()
    {
        return fYAxis;
    }

}

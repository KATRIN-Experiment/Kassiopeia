#include "KSMathPredictorCorrector.h"
#include "KSMathMessage.h"

#include "TMath.h"


namespace Kassiopeia
{
    
    KSMathPredictorCorrector::KSMathPredictorCorrector() :
        fDimension(0),
        fCount(0),
        fSum(0.0),
        fPredictionConditionBlock(0),
        fCorrectionConditionBlock(0),
        fDerivativeBlockAddress(0)
    {
        //initialize the array of derivatives to zero
        for( UInt_t i = 0; i < ePrevStepCount; i++ )
        {
            fDerivatives[i] = 0;
        }
        
        //set the iterator to the derivative array
        fIter.SetArray(fDerivatives,ePrevStepCount);
    }
    
    KSMathPredictorCorrector::~KSMathPredictorCorrector()
    {
        if( fDerivativeBlockAddress != 0 )
        {
            delete[] fDerivativeBlockAddress;
            fDerivativeBlockAddress = 0;
        }
        if( fPredictionConditionBlock != 0 )
        {
            delete[] fPredictionConditionBlock;
            fPredictionConditionBlock = 0;
        }
        if( fCorrectionConditionBlock != 0 )
        {
            delete[] fCorrectionConditionBlock;
            fCorrectionConditionBlock = 0;
        }
    }
    
    void KSMathPredictorCorrector::SolveAction()
    {
        //if our previous timestep suggestion was not followed, we must start all over with calculation of back values
        UInt_t iSolution, iPastStep;        

        if( fTimeStep != fNewTimeStep )
        {
            fCount = 0;
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SolveAction";
            mathmsg(eDebug) << "KMathPredictorCorrector::Solve: given timestep triggered reset" << eom;
#endif            

        }

#ifdef DEBUG_VERBOSE
        mathmsg < "KMathPredictorCorrector::SolveAction";
        mathmsg(eDebug) << "initial condition: ";
        mathmsg << "[";
        for( UInt_t index = 0; index < fDimension; index++ )
        {
            mathmsg << fInitialConditionBlock[index] << ", ";
        }
        mathmsg << "]" << eom;
#endif

        //save the derivative at the initial point to the marked position in the cyclic iterator
        fIter.ToMark();
        fODE->SetFunctionBlock(fInitialConditionBlock);
        fODE->SetDerivativeBlock((*fIter));
        fODE->Evaluate();
        
#ifdef DEBUG_VERBOSE
        mathmsg < "KMathPredictorCorrector::SolveAction";
        mathmsg(eDebug) << "initial derivative: ";
        mathmsg << "[";
        for( UInt_t index = 0; index < fDimension; index++ )
        {
            mathmsg << (*fIter)[index] << ", ";
        }
        mathmsg << "]" << eom;
#endif        
        
        //first 7 steps are solved by rk8
        if( fCount < 7 )
        {   
            //get the solution from rk8
            fRK8.SetInitialConditionBlock(fInitialConditionBlock);
            fRK8.SetFinalConditionBlock(fFinalConditionBlock);
            fRK8.Solve(fTimeStep);             
            
            //advance the iterator mark
            +fIter;
            //if the back value count is less than the space we have allotted, increment the count
            if( fCount < 14 )
            {
                fCount++;
            }
        }
        //subsequent steps are solved by predictor-corrector
        else
        {        
            //use the Adams-Bashforth predictor to estimate the next step
            for( iSolution = 0; iSolution < fDimension; iSolution++ )
            {
                fSum = 0.0;
                fIter.ToMark();
                for( iPastStep = 0; iPastStep < 8; iPastStep++ )
                {
                    fSum += fP[iPastStep] * (*fIter)[iSolution];
                    --fIter;
                }
                fPredictionConditionBlock[iSolution] = fInitialConditionBlock[iSolution] + fSum * fTimeStep;
            }

            //advance the iterator mark
            ++fIter;
            //if the back value count is less than the space we have allotted, increment the count
            if( fCount < 14 )
            {
                fCount++;
            }            
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SolveAction";
            mathmsg(eDebug) << "prediction: ";
            mathmsg << "[";
            for( UInt_t index = 0; index < fDimension; index++ )
            {
                mathmsg << fPredictionConditionBlock[index] << ", ";
            }
            mathmsg << "]" << eom;
#endif        
            
            //save the derivative at the predicted new point to the marked position in the cyclic iterator
            fIter.ToMark();
            fODE->SetFunctionBlock(fPredictionConditionBlock);
            fODE->SetDerivativeBlock(*fIter);
            fODE->Evaluate();
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SolveAction";
            mathmsg(eDebug) << "derivative at prediction: ";
            mathmsg << "[";
            for( UInt_t index = 0; index < fDimension; index++ )
            {
                mathmsg << (*fIter)[index] << ", ";
            }
            mathmsg << "]" << eom;
#endif        
            
            for( iSolution = 0; iSolution < fDimension; iSolution++ )
            {
                fSum = 0.0;
                fIter.ToMark();
                for( iPastStep = 0; iPastStep < 8; iPastStep++ )
                {
                    fSum += fC[iPastStep] * (*fIter)[iSolution];
                    --fIter;
                }
                fCorrectionConditionBlock[iSolution] = (33953./1103970.)*fPredictionConditionBlock[iSolution] + (1070017./1103970.)*( fInitialConditionBlock[iSolution] + fSum * fTimeStep );
            }
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SolveAction";
            mathmsg(eDebug) << "modified correction: ";
            mathmsg << "[";
            for( UInt_t index = 0; index < fDimension; index++ )
            {
                mathmsg << fCorrectionConditionBlock[index] << ", ";
            }
            mathmsg << "]" << eom;
#endif

            //save the derivative at the corrected new point to the marked position in the cyclic iterator
            fIter.ToMark();
            fODE->SetFunctionBlock(fCorrectionConditionBlock);
            fODE->SetDerivativeBlock(*fIter);
            fODE->Evaluate();
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SolveAction";
            mathmsg(eDebug) << "derivative at modified correction: ";
            mathmsg << "[";
            for( UInt_t index = 0; index < fDimension; index++ )
            {
                mathmsg << (*fIter)[index] << ", ";
            }
            mathmsg << "]" << eom;
#endif

            for( iSolution = 0; iSolution < fDimension; iSolution++ )
            {
                fSum = 0.0;
                fIter.ToMark();
                for( iPastStep = 0; iPastStep < 8; iPastStep++ )
                {
                    fSum += fC[iPastStep] * (*fIter)[iSolution];
                    --fIter;
                }
                fFinalConditionBlock[iSolution] = fInitialConditionBlock[iSolution] + fSum * fTimeStep;
            }
        } 
        
#ifdef DEBUG_VERBOSE
        mathmsg < "KMathPredictorCorrector::SolveAction";
        mathmsg(eDebug) << "solution: ";
        mathmsg << "[";
        for( UInt_t index = 0; index < fDimension; index++ )
        {
            mathmsg << fFinalConditionBlock[index] << ", ";
        }
        mathmsg << "]" << ret;
        mathmsg << "back values:";
        mathmsg << "[" << fCount << "]" << eom;
#endif 
        
        //the default next time step suggestion is the same as the one just taken
        fNewTimeStep = fTimeStep;
        
        return;
    }
    
    void KSMathPredictorCorrector::CheckAction()
    {   
        //if we don't have enough back values, we can't check anything
        if( fCount < 7 )
        {
            fValidStep = true;
            return;
        }
        
        UInt_t iSolution;
        Double_t Ratio;
        Double_t Error;
        Double_t MinErrRatio = 0.0;
        Double_t MaxErrRatio = 0.0;
        
        for( iSolution = 0; iSolution < fDimension; iSolution++ )
        {
            Error = fFinalConditionBlock[iSolution] - fPredictionConditionBlock[iSolution];

            Ratio = TMath::Abs(Error/fMaxErrorConditionBlock[iSolution]);
            if( Ratio > MaxErrRatio )
            {
                MaxErrRatio = Ratio;
            }
            
            Ratio = TMath::Abs(Error/fMinErrorConditionBlock[iSolution]);
            if( Ratio > MinErrRatio )
            {
                MinErrRatio = Ratio;
            }
        }
        
        if( MaxErrRatio > 1.0 )
        {
            fValidStep = false;
            HalveTimeStep();
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::CheckAction";
            mathmsg(eDebug) << "step too large: ";
            mathmsg << "[" << MaxErrRatio << "]" << ret;
            mathmsg << "sugg. timestep: ";
            mathmsg << fNewTimeStep << eom;
#endif
            
            return;
        }

        fValidStep = true;
        
        if( MinErrRatio < 1.0 )
        {
            if( fCount == 14 )
            {
                DoubleTimeStep();
            }
            
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::CheckAction";
            mathmsg(eDebug) << "step too small";
            mathmsg << "[" << MinErrRatio << "]" << ret;
            mathmsg << "sugg. timestep:";
            mathmsg << fNewTimeStep << eom;
#endif
            
            return;
        }
        
        return;
    }
    
    void KSMathPredictorCorrector::ComputeTimeStepAction()
    {
        return;
    }
    
    void KSMathPredictorCorrector::SetODEAction()
    {
        //set the internal rk8 to point to the new ode and clear the count of back steps
#ifdef DEBUG_VERBOSE
        mathmsg < "KMathPredictorCorrector::SetODEAction";
        mathmsg(eDebug) << "setting internal RK8 and clearing back count" << eom;
#endif        
        fRK8.SetODE(fODE);        
        fCount = 0;
        
        //locally grab the dimension of the new ode
        UInt_t NewDimension = fODE->GetDimension();
        
#ifdef DEBUG_VERBOSE
        mathmsg < "KMathPredictorCorrector::SetODEAction";
        mathmsg(eDebug) << "got a new ode of dimension " << NewDimension << eom;
#endif

        //check to see if we need to resize our arrays of derivatives and intermediate conditions
        if( NewDimension != fDimension )
        {
            fDimension = NewDimension;
            
            //release previously allocated dervative array if necessary
            if( fDerivativeBlockAddress != 0 )
            {
#ifdef DEBUG_VERBOSE
                mathmsg < "KMathPredictorCorrector::SetODEAction";
                mathmsg(eDebug) << "deleting old derivative array at " << fDerivatives[0] << eom;
#endif
                delete[] fDerivativeBlockAddress;
                fDerivativeBlockAddress = 0;
            }
            
            //release previously allocated prediction condition array if necessary
            if( fPredictionConditionBlock != 0 )
            {
#ifdef DEBUG_VERBOSE
                mathmsg < "KMathPredictorCorrector::SetODEAction";
                mathmsg(eDebug) << "deleting old prediction condition array at " << fPredictionConditionBlock << eom;
#endif
                delete[] fPredictionConditionBlock;
                fPredictionConditionBlock = 0;
            }
            
            //release previously allocated truncation error array if necessary
            if( fCorrectionConditionBlock != 0 )
            {
#ifdef DEBUG_VERBOSE
                mathmsg < "KMathPredictorCorrector::SetODEAction";
                mathmsg(eDebug) << "deleting old correction condition array at " << fCorrectionConditionBlock << eom;
#endif
                delete[] fCorrectionConditionBlock;
                fCorrectionConditionBlock = 0;
            }
            
            //allocate new memory for the derivative array
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SetODEAction";
            mathmsg(eDebug) << "allocating new derivative array of dimension " << ePrevStepCount*fDimension << " and setting iterator " << eom;
#endif
            fDerivatives[0] = new Double_t[ePrevStepCount*fDimension]();
            fDerivativeBlockAddress = fDerivatives[0];
            for( UInt_t i = 1; i < ePrevStepCount; i++ )
            {
                fDerivatives[i] = fDerivatives[i-1] + fDimension;
            }
            fIter.SetArray(fDerivatives,ePrevStepCount);            
            
            //allocate new memory for the intermediate condition array
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SetODEAction";
            mathmsg(eDebug) << "allocating new prediction condition array of dimension " << fDimension << eom;
#endif
            fPredictionConditionBlock = new Double_t[fDimension]();
            
            //allocate new memory for the truncation error array
#ifdef DEBUG_VERBOSE
            mathmsg < "KMathPredictorCorrector::SetODEAction";
            mathmsg(eDebug) << "allocating new correction condition array of dimension " << fDimension << eom;
#endif
            fCorrectionConditionBlock = new Double_t[fDimension]();

        }
        
        return;        
    }
    
    void KSMathPredictorCorrector::ResetAction()
    {
        fCount = 0;
        /*
        fIter.ToMark();
        do
        {
            for( UInt_t iSolution = 0; iSolution < fDimension; iSolution++ )
            {
                (*fIter)[iSolution] = 0.0;
            }
        }
        while(++fIter != false );
        */
        return;
    }
    
    void KSMathPredictorCorrector::DoubleTimeStep()
    {
        fNewTimeStep = 2.0 * fTimeStep;
        fCount = 7;
        
        fIter.ToMark();
        KSCyclicIterator< Double_t* > From(fIter);
        KSCyclicIterator< Double_t* > To(fIter);
                
        for( UInt_t index = 0; index < 7; index++ )
        {
            From.IncrementPosition(2);
            ++To;
            (*To) = (*From);
        }
        
        return;
    }
    
    void KSMathPredictorCorrector::HalveTimeStep()
    {
        fNewTimeStep = 0.5 * fTimeStep;
        fCount = 8;
        
        fIter.ToMark();
        KSCyclicIterator< Double_t* > From(fIter);
        From.IncrementPosition(7);
        KSCyclicIterator< Double_t* > To(fIter);
        To.IncrementPosition(14);
        
        for( UInt_t index = 0; index < 7; index++ )
        {
            (*To) = (*From);
            --From;
            To.IncrementPosition(-2);
        }
        
        for( UInt_t iSolution = 0; iSolution < fDimension; iSolution++ )
        {
            To.ToMark();
            for( UInt_t iTo = 0; iTo < 4; iTo++ )
            {
                ++To;
                fSum = 0.0;                
                From.ToMark();
                for( UInt_t iFrom = 0; iFrom < 8; iFrom++ )
                {
                    fSum += fInterpolationCoeffs[iTo][iFrom] * (*From)[iSolution];
                    ++From;
                    ++From;
                }
                (*To)[iSolution] = fSum;
                ++To;
            }
        }
        
        return;
    }
    
    
    //predictor coefficients
    const Double_t KSMathPredictorCorrector::fP[8] =
	{16083./4480.,-1152169./120960.,242653./13440.,-296053./13440.,2102243./120960.,-115747./13440.,32863./13440.,-5257./17280.};

    //corrector coefficients
    const Double_t KSMathPredictorCorrector::fC[8] =
	{5257./17280.,139849./120960.,-4511./4480.,123133./120960.,-88547./120960.,1537./4480.,-11351./120960.,275./24192.};
	
	const Double_t KSMathPredictorCorrector::fInterpolationCoeffs[4][8]=
	{	
		{135135./645120.,135135./92160.,-45045./30720.,27027./18432.,-19305./18432.,15015./30720.,-12285./92160.,10395./645120.},
		{-10395./645120.,31185./92160.,31185./30720.,-10395./18432.,6237./18432.,-4455./30720.,3465./92160.,-2835./645120.},
		{2835./645120.,-4725./92160.,14175./30720.,14175./18432.,-4725./18432.,2835./30720.,-2025./92160.,1575./645120.},
		{-1575./645120.,2205./92160.,-3675./30720.,11025./18432.,11025./18432.,-3675./30720.,2205./92160.,-1575./645120.}
	};
}

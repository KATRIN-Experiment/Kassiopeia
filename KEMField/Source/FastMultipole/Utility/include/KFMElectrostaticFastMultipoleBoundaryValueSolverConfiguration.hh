#ifndef KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration_HH__
#define KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration_HH__

#include <string>
#include <limits.h>
#include "KSAStructuredASCIIHeaders.hh"
#include "KFMElectrostaticParameters.hh"
#include "KKrylovSolverConfiguration.hh"

namespace KEMField
{

/*
*
*@file KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration.hh
*@class KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Feb 8 14:16:47 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration: public KSAInputOutputObject
{
    public:

        KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration()
    :fFFTMParams(NULL),fPreconditionerFFTMParams(NULL)
        {
            fPreconditionerName = "none";

            fPreconditionerTolerance = 0.1;
            fMaxPreconditionerIterations = UINT_MAX;
            fPreconditionerDegree = 0;
        }

        virtual ~KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration();

        std::string GetSolverName() const {return fSolverParams.GetSolverName();};
        void SetSolverName(const std::string& name){fSolverParams.SetSolverName(name);};

        std::string GetPreconditionerName() const {return fPreconditionerName;};
        void SetPreconditionerName(const std::string& n){fPreconditionerName = n;};

        double GetSolverTolerance() const {return fSolverParams.GetTolerance();};
        void SetSolverTolerance(const double& t){fSolverParams.SetTolerance(t);};

        int GetMaxSolverIterations() const {return fSolverParams.GetMaxIterations();};
        void SetMaxSolverIterations(const int& n){fSolverParams.SetMaxIterations(n);};

        int GetIterationsBetweenRestart() const {return fSolverParams.GetIterationsBetweenRestart();};
        void SetIterationsBetweenRestart(const int& d){fSolverParams.SetIterationsBetweenRestart(d);};

        double GetPreconditionerTolerance() const {return fPreconditionerTolerance;};
        void SetPreconditionerTolerance(const double& t){fPreconditionerTolerance = t;};

        int GetMaxPreconditionerIterations() const {return fMaxPreconditionerIterations;};
        void SetMaxPreconditionerIterations(const int& n){fMaxPreconditionerIterations = n;};

        int GetPreconditionerDegree() const {return fPreconditionerDegree;};
        void SetPreconditionerDegree(const int& deg){fPreconditionerDegree = deg;};

        int GetUseCheckpoints() const {return fSolverParams.IsUseCheckpoints();};
        void SetUseCheckpoints(const int& z){fSolverParams.SetUseCheckpoints(z);};

        int GetCheckpointFrequency() const {return fSolverParams.GetStepsBetweenCheckpoints();};
        void SetCheckpointFrequency(const int& z){fSolverParams.SetStepsBetweenCheckpoints(z);};

        int GetUseDisplay() const {return fSolverParams.IsUseDisplay();};
        void SetUseDisplay(const int& t){fSolverParams.SetUseDisplay(t);};

        int GetUsePlot() const {return fSolverParams.IsUsePlot();};
        void SetUsePlot(const int& r){fSolverParams.SetUsePlot(r);};

        int GetUseTimer() const {return fSolverParams.IsUseTimer();};
        void SetUseTimer(const int& r){fSolverParams.SetUseTimer(r);};

        double GetTimeLimitSeconds() const {return fSolverParams.GetTimeLimitSeconds();};
        void SetTimeLimitSeconds(const double& t){fSolverParams.SetTimeLimitSeconds(t);};

        int GetTimeCheckFrequency() const {return fSolverParams.GetStepsBetweenTimeChecks();};
        void SetTimeCheckFrequency(const int& f){fSolverParams.SetStepsBetweenTimeChecks(f);};

        KFMElectrostaticParameters* GetFFTMParams(){return fFFTMParams;}
        void SetFFTMParams(KFMElectrostaticParameters* config);

        KFMElectrostaticParameters* GetPreconditionerFFTMParams()
        	{return fPreconditionerFFTMParams;}
        void SetPreconditionerFFTMParams(KFMElectrostaticParameters* config);


        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,SolverName,std::string);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,PreconditionerName,std::string);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,SolverTolerance,double);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,PreconditionerTolerance,double);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,MaxSolverIterations,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,MaxPreconditionerIterations,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,IterationsBetweenRestart,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,PreconditionerDegree,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UseCheckpoints,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,CheckpointFrequency,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UseDisplay,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UsePlot,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UseTimer,int);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,TimeLimitSeconds,double);
            AddKSAOutputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,TimeCheckFrequency,int);
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,SolverName,std::string);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,PreconditionerName,std::string);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,SolverTolerance,double);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,PreconditionerTolerance,double);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,MaxSolverIterations,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,MaxPreconditionerIterations,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,IterationsBetweenRestart,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,PreconditionerDegree,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UseCheckpoints,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,CheckpointFrequency,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UseDisplay,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UsePlot,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,UseTimer,int);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,TimeLimitSeconds,double);
            AddKSAInputFor(KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration,TimeCheckFrequency,int);
        }

        virtual std::string ClassName() const {return std::string("KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration");};

    protected:

        std::string fPreconditionerName;

        double fPreconditionerTolerance;
        int fMaxPreconditionerIterations;
        int fPreconditionerDegree;

        KKrylovSolverConfiguration fSolverParams;
        KFMElectrostaticParameters* fFFTMParams;
        KFMElectrostaticParameters* fPreconditionerFFTMParams;
};

DefineKSAClassName( KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration );

}

#endif /* KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration_HH__ */

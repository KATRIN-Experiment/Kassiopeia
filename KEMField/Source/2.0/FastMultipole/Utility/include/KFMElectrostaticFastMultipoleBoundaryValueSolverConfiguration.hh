#ifndef KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration_HH__
#define KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration_HH__

#include <string>
#include <limits.h>
#include "KSAStructuredASCIIHeaders.hh"
#include "KFMElectrostaticParameters.hh"

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
            fSolverName = "gmres";
            fPreconditionerName = "none";

            fSolverTolerance = 0.1;
            fMaxSolverIterations = UINT_MAX;
            fIterationsBetweenRestart = UINT_MAX;
            fPreconditionerTolerance = 0.1;
            fMaxPreconditionerIterations = UINT_MAX;
            fPreconditionerDegree = 0;

            fUseCheckpoints = 0;
            fCheckpointFrequency = 1;

            fUseDisplay = 0;
            fUsePlot = 0;

            fUseTimer = 0;
            fTimeLimitSeconds = 3e10; //seconds
            fTimeCheckFrequency = 1;
        }

        virtual ~KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration();

        std::string GetSolverName() const {return fSolverName;};
        void SetSolverName(const std::string& name){fSolverName = name;};

        std::string GetPreconditionerName() const {return fPreconditionerName;};
        void SetPreconditionerName(const std::string& n){fPreconditionerName = n;};

        double GetSolverTolerance() const {return fSolverTolerance;};
        void SetSolverTolerance(const double& t){fSolverTolerance = t;};

        int GetMaxSolverIterations() const {return fMaxSolverIterations;};
        void SetMaxSolverIterations(const int& n){fMaxSolverIterations = n;};

        int GetIterationsBetweenRestart() const {return fIterationsBetweenRestart;};
        void SetIterationsBetweenRestart(const int& d){fIterationsBetweenRestart = d;};

        double GetPreconditionerTolerance() const {return fPreconditionerTolerance;};
        void SetPreconditionerTolerance(const double& t){fPreconditionerTolerance = t;};

        int GetMaxPreconditionerIterations() const {return fMaxPreconditionerIterations;};
        void SetMaxPreconditionerIterations(const int& n){fMaxPreconditionerIterations = n;};

        int GetPreconditionerDegree() const {return fPreconditionerDegree;};
        void SetPreconditionerDegree(const int& deg){fPreconditionerDegree = deg;};

        int GetUseCheckpoints() const {return fUseCheckpoints;};
        void SetUseCheckpoints(const int& z){fUseCheckpoints = z;};

        int GetCheckpointFrequency() const {return fCheckpointFrequency;};
        void SetCheckpointFrequency(const int& z){fCheckpointFrequency = z;};

        int GetUseDisplay() const {return fUseDisplay;};
        void SetUseDisplay(const int& t){fUseDisplay = t;};

        int GetUsePlot() const {return fUsePlot;};
        void SetUsePlot(const int& r){fUsePlot= r;};

        int GetUseTimer() const {return fUseTimer;};
        void SetUseTimer(const int& r){fUseTimer = r;};

        double GetTimeLimitSeconds() const {return fTimeLimitSeconds;};
        void SetTimeLimitSeconds(const double& t){fTimeLimitSeconds = t;};

        int GetTimeCheckFrequency() const {return fTimeCheckFrequency;};
        void SetTimeCheckFrequency(const int& f){fTimeCheckFrequency = f;};

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

        std::string fSolverName;
        std::string fPreconditionerName;

        double fSolverTolerance;
        int fMaxSolverIterations;
        int fIterationsBetweenRestart;
        double fPreconditionerTolerance;
        int fMaxPreconditionerIterations;
        int fPreconditionerDegree;

        int fUseCheckpoints;
        int fCheckpointFrequency;

        int fUseDisplay;
        int fUsePlot;

        int fUseTimer;
        double fTimeLimitSeconds;
        int fTimeCheckFrequency;

        KFMElectrostaticParameters* fFFTMParams;
        KFMElectrostaticParameters* fPreconditionerFFTMParams;
};

DefineKSAClassName( KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration );

}

#endif /* KFMElectrostaticFastMultipoleBoundaryValueSolverConfiguration_HH__ */

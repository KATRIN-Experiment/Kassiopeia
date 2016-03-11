#ifndef KFMWorkLoadBalanceWeights_HH__
#define KFMWorkLoadBalanceWeights_HH__

#include "KSAStructuredASCIIHeaders.hh"

namespace KEMField
{

/*
*
*@file KFMWorkLoadBalanceWeights.hh
*@class KFMWorkLoadBalanceWeights
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sun Feb 8 14:16:47 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

class KFMWorkLoadBalanceWeights: public KSAInputOutputObject
{
    public:

        KFMWorkLoadBalanceWeights()
        {
            fDivisions = 0;
            fZeroMaskSize = 0;
            fRamMatrixVectorProductWeight = 0;
            fDiskMatrixVectorProductWeight = 0;
            fFFTWeight = 0;
        }

        virtual ~KFMWorkLoadBalanceWeights(){;};

        unsigned int GetDivisions() const {return fDivisions;};
        void SetDivisions(const unsigned int& d){fDivisions = d;};

        unsigned int GetZeroMaskSize() const {return fZeroMaskSize;};
        void SetZeroMaskSize(const unsigned int& z){fZeroMaskSize = z;};

        double GetDiskMatrixVectorProductWeight() const {return fDiskMatrixVectorProductWeight;};
        void SetDiskMatrixVectorProductWeight(const double& d){fDiskMatrixVectorProductWeight = d;};

        double GetRamMatrixVectorProductWeight() const {return fRamMatrixVectorProductWeight;};
        void SetRamMatrixVectorProductWeight(const double& d){fRamMatrixVectorProductWeight = d;};

        double GetFFTWeight() const {return fFFTWeight;};
        void SetFFTWeight(const double& d){fFFTWeight = d;};

        void DefineOutputNode(KSAOutputNode* node) const
        {
            AddKSAOutputFor(KFMWorkLoadBalanceWeights,Divisions,unsigned int);
            AddKSAOutputFor(KFMWorkLoadBalanceWeights,ZeroMaskSize,unsigned int);
            AddKSAOutputFor(KFMWorkLoadBalanceWeights,DiskMatrixVectorProductWeight,double);
            AddKSAOutputFor(KFMWorkLoadBalanceWeights,RamMatrixVectorProductWeight,double);
            AddKSAOutputFor(KFMWorkLoadBalanceWeights,FFTWeight,double);
        }

        void DefineInputNode(KSAInputNode* node)
        {
            AddKSAInputFor(KFMWorkLoadBalanceWeights,Divisions,unsigned int);
            AddKSAInputFor(KFMWorkLoadBalanceWeights,ZeroMaskSize,unsigned int);
            AddKSAInputFor(KFMWorkLoadBalanceWeights,DiskMatrixVectorProductWeight,double);
            AddKSAInputFor(KFMWorkLoadBalanceWeights,RamMatrixVectorProductWeight,double);
            AddKSAInputFor(KFMWorkLoadBalanceWeights,FFTWeight,double);
        }

        virtual std::string ClassName() const {return std::string("KFMWorkLoadBalanceWeights");};

    protected:

        unsigned int fDivisions;
        unsigned int fZeroMaskSize;
        double fRamMatrixVectorProductWeight;
        double fDiskMatrixVectorProductWeight;
        double fFFTWeight;


};

DefineKSAClassName( KFMWorkLoadBalanceWeights );

}

#endif /* KFMWorkLoadBalanceWeights_HH__ */

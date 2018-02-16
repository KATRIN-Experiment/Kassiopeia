#ifndef KFMMomentTransformer_H__
#define KFMMomentTransformer_H__


#include <cmath>
#include <cstddef>
#include <vector>
#include <complex>

namespace KEMField{

/**
*
*@file KFMMomentTransformer.hh
*@class KFMMomentTransformer
*@brief This is a slow but general class for moment to moment transformation
* It should primarily be used in intialization or for testing, as any direct repeated use
* will be very slow compared to a more customized method
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Oct  2 18:56:02 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


template <class KernelType> //KernelType must inherit from KFMKernelExpansion
class KFMMomentTransformer
{
    public:
        KFMMomentTransformer(){;}
        virtual ~KFMMomentTransformer(){;};

        virtual void Initialize(){fKernel.Initialize();};

        virtual void SetSourceOrigin(double* origin){fKernel.SetSourceOrigin(origin);};
        virtual void SetTargetOrigin(double* origin){fKernel.SetTargetOrigin(origin);};

        virtual void SetNumberOfTermsInSeries(unsigned int n_terms)
        {
            fNTerms = n_terms;
            fSourceMoments.clear();
            fSourceMoments.resize(fNTerms);
            fTargetMoments.clear();
            fTargetMoments.resize(fNTerms);
        };

        virtual void SetSourceMoments(const std::vector< std::complex<double> >* moments)
        {
            fSourceMoments = *moments;
        };

        virtual void GetTargetMoments(std::vector<std::complex<double> >* moments) const
        {
            *moments = fTargetMoments;
        };

        virtual void Transform()
        {
            std::complex<double> result;
            for(unsigned int target = 0; target < fNTerms; target++)
            {
                result = std::complex<double>(0,0);
                for(unsigned int source = 0; source < fNTerms; source++)
                {
                    if(fKernel.IsPhysical(source, target) )
                    {
                        result += (fSourceMoments[source])*(fKernel.GetResponseFunction(source, target));
                    }
                }
                fTargetMoments[target] = result;
            }
        }

        //transform given an external set of response functions
        virtual void Transform(std::vector<std::complex<double> >* response_functions)
        {
            //expects the response functions to be indexed as [target][source] -> (source + target*n_terms)
            std::complex<double> result;
            unsigned int r_index;
            for(unsigned int target = 0; target < fNTerms; target++)
            {
                result = std::complex<double>(0,0);
                for(unsigned int source = 0; source < fNTerms; source++)
                {
                    r_index = source + target*fNTerms;
                    result += (fSourceMoments[source])*((*response_functions)[r_index]);
                    fTargetMoments[target] = result;
                }
            }
        }

    protected:

        KernelType fKernel;
        unsigned int fNTerms;

        std::vector<std::complex<double> > fSourceMoments;
        std::vector<std::complex<double> > fTargetMoments;

};

}//end of KEMField namespace


#endif /* __KFMMomentTransformer_H__ */

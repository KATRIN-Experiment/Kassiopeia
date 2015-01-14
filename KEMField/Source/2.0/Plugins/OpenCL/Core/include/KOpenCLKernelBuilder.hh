#ifndef KOpenCLKernelBuilder_HH__
#define KOpenCLKernelBuilder_HH__

#include "KOpenCLInterface.hh"
#include <string>

namespace KEMField
{

class KOpenCLKernelBuilder
{
    public:
        KOpenCLKernelBuilder(){};
        virtual ~KOpenCLKernelBuilder(){};

        cl::Kernel* BuildKernel(std::string SourceFileName, std::string KernelName, std::string BuildFlags=std::string("") );

    protected:

};

}

#endif /* KOpenCLKernelBuilder_H__ */

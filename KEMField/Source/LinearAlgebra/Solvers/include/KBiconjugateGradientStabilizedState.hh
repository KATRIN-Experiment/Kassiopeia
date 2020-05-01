#ifndef __KBiconjugateGradientStabilizedState_H__
#define __KBiconjugateGradientStabilizedState_H__

#include "KSimpleVector.hh"

#include <string>

#ifdef KEMFIELD_USE_MPI
#include "KMPIInterface.hh"
#ifndef MPI_ROOT_PROCESS_ONLY
#define MPI_ROOT_PROCESS_ONLY if (KMPIInterface::GetInstance()->GetProcess() == 0)
#endif
#else
#ifndef MPI_ROOT_PROCESS_ONLY
#define MPI_ROOT_PROCESS_ONLY
#endif
#endif

namespace KEMField
{

/**
*
*@file KBiconjugateGradientStabilizedState.hh
*@class KBiconjugateGradientStabilizedState
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 14 10:16:13 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType> class KBiconjugateGradientStabilizedState
{
  public:
    typedef KSimpleVector<ValueType> KSimpleVectorType;

    KBiconjugateGradientStabilizedState(){};
    virtual ~KBiconjugateGradientStabilizedState(){};

    static std::string Name()
    {
        return std::string("bicgstab");
    }
    std::string NameLabel()
    {
        return std::string("bicgstab");
    }

    virtual void SetDimension(unsigned int dim)
    {
        fDim = dim;
    };
    virtual unsigned int GetDimension() const
    {
        return fDim;
    };

    virtual const KSimpleVector<ValueType>* GetSolutionVector() const
    {
        return &fX;
    };
    virtual void SetSolutionVector(const KSimpleVector<ValueType>* x)
    {
        unsigned int size = x->size();
        fX.resize(size);
        for (unsigned int i = 0; i < size; i++) {
            fX[i] = (*x)(i);
        };
    };

    virtual const KSimpleVector<ValueType>* GetRightHandSide() const
    {
        return &fB;
    };
    virtual void SetRightHandSide(const KSimpleVector<ValueType>* b)
    {
        unsigned int size = b->size();
        fB.resize(size);
        for (unsigned int i = 0; i < size; i++) {
            fB[i] = (*b)(i);
        };
    };

    virtual void SynchronizeData()
    {
//now broadcast the data to all of the other processes
//IMPORTANT!! THIS ONLY WORKS FOR ValueType == double
//Other types are not yet implemented for MPI version of this code
#ifdef KEMFIELD_USE_MPI

        unsigned int x_size = fX.size();
        unsigned int b_size = fB.size();

        const int root_id = 0;
        int proc_id = KMPIInterface::GetInstance()->GetProcess();

        KMPIInterface::GetInstance()->GlobalBarrier();

        MPI_Bcast(&(fDim), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD);
        MPI_Bcast(&(x_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD);
        MPI_Bcast(&(b_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD);

        if (proc_id != root_id) {
            fX.resize(x_size);
            fB.resize(b_size);
        }

        KMPIInterface::GetInstance()->GlobalBarrier();

        MPI_Bcast(&(fX[0]), x_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
        MPI_Bcast(&(fB[0]), b_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);

#endif
    }

  protected:
    unsigned int fDim;
    //solution vector
    KSimpleVector<ValueType> fX;
    //right hand side
    KSimpleVector<ValueType> fB;
};


template<typename ValueType, typename Stream>
Stream& operator>>(Stream& s, KBiconjugateGradientStabilizedState<ValueType>& aData)
{
    s.PreStreamInAction(aData);

    unsigned int dim;
    unsigned int x_size;
    unsigned int b_size;

    KSimpleVector<ValueType> x;
    KSimpleVector<ValueType> b;

    s >> dim;
    s >> x_size;
    for (unsigned int i = 0; i < x_size; i++) {
        ValueType temp;
        s >> temp;
        x.push_back(temp);
    }

    s >> b_size;
    for (unsigned int i = 0; i < b_size; i++) {
        ValueType temp;
        s >> temp;
        b.push_back(temp);
    }

    aData.SetDimension(dim);
    aData.SetSolutionVector(&x);
    aData.SetRightHandSide(&b);

    s.PostStreamInAction(aData);
    return s;
}


template<typename ValueType, typename Stream>
Stream& operator<<(Stream& s, const KBiconjugateGradientStabilizedState<ValueType>& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetDimension();

    const KSimpleVector<ValueType>* x = aData.GetSolutionVector();
    unsigned int x_size = x->size();
    s << x_size;
    for (unsigned int i = 0; i < x_size; i++) {
        s << (*x)(i);
    }

    const KSimpleVector<ValueType>* b = aData.GetRightHandSide();
    unsigned int b_size = b->size();
    s << b_size;
    for (unsigned int i = 0; i < b_size; i++) {
        s << (*b)(i);
    }

    s.PostStreamOutAction(aData);

    return s;
}


}  // namespace KEMField

#endif /* __KBiconjugateGradientStabilizedState_H__ */

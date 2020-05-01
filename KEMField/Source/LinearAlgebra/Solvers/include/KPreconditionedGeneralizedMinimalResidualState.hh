#ifndef __KPreconditionedGeneralizedMinimalResidualState_H__
#define __KPreconditionedGeneralizedMinimalResidualState_H__

#include "KGeneralizedMinimalResidualState.hh"

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
*@file KPreconditionedGeneralizedMinimalResidualState.hh
*@class KPreconditionedGeneralizedMinimalResidualState
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Thu Jan 15 20:08:38 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<typename ValueType>
class KPreconditionedGeneralizedMinimalResidualState : public KGeneralizedMinimalResidualState<ValueType>
{
  public:
    typedef KSimpleVector<ValueType> KSimpleVectorType;

    KPreconditionedGeneralizedMinimalResidualState() : KGeneralizedMinimalResidualState<ValueType>(){};
    ~KPreconditionedGeneralizedMinimalResidualState() override{};

    static std::string Name()
    {
        return std::string("pgmres");
    }
    std::string NameLabel()
    {
        return std::string("pgmres");
    }

    const std::vector<KSimpleVector<ValueType>>* GetPreconditionedKrylovSpaceBasis() const
    {
        return &fZ;
    };

    void GetPreconditionedKrylovSpaceBasis(std::vector<KSimpleVector<ValueType>>* h) const
    {
        h->clear();
        unsigned int s = fZ.size();
        h->resize(s);

        for (unsigned int i = 0; i < s; i++) {
            unsigned int t = (fZ.at(i)).size();
            (*h)[i].resize(t);
            for (unsigned int j = 0; j < t; j++) {
                (*h)[i][j] = fZ[i](j);
            }
        }
    }


    void SetPreconditionedKrylovSpaceBasis(const std::vector<KSimpleVector<ValueType>>* z)
    {
        if (fZ.size() != 0) {
            for (unsigned int i = 0; i < fZ.size(); i++) {
                fZ[i].clear();
            };
        };
        fZ.clear();

        unsigned int s = z->size();
        fZ.resize(s);

        for (unsigned int i = 0; i < s; i++) {
            unsigned int t = (z->at(i)).size();
            fZ[i].resize(t);
            for (unsigned int j = 0; j < t; j++) {
                fZ[i][j] = (z->at(i))(j);
            }
        }
    };

    void SynchronizeData() override
    {

        KGeneralizedMinimalResidualState<ValueType>::SynchronizeData();
        /*
            //now broadcast the data to all of the other processes
            //IMPORTANT!! THIS ONLY WORKS FOR ValueType == double
            //Other types are not yet implemented for MPI version of this code
            #ifdef KEMFIELD_USE_MPI

            unsigned int z_row_size = fZ.size();

            KMPIInterface::GetInstance()->GlobalBarrier();
            const int root_id = 0;
            int proc_id = KMPIInterface::GetInstance()->GetProcess();

            MPI_Bcast( &(z_row_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );

            if(proc_id != root_id)
            {
                fZ.resize(z_row_size);
            }

            KMPIInterface::GetInstance()->GlobalBarrier();

            for(unsigned int i=0; i<z_row_size; i++)
            {
                unsigned int z_col_size;
                if(proc_id == root_id)
                {
                    z_col_size = fZ[i].size();
                }

                MPI_Bcast( &(z_col_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD);

                if(proc_id != root_id)
                {
                    fZ[i].resize(z_col_size);
                }

                MPI_Bcast( &(fZ[i][0]), z_col_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            }

            #endif
*/
    }

  protected:
    //the preconditioned krylov subspace basis vectors
    std::vector<KSimpleVector<ValueType>> fZ;
};


template<typename ValueType, typename Stream>
Stream& operator>>(Stream& s, KPreconditionedGeneralizedMinimalResidualState<ValueType>& aData)
{
    s.PreStreamInAction(aData);

    auto* base = static_cast<KGeneralizedMinimalResidualState<ValueType>*>(&(aData));
    s >> *base;

    std::vector<KSimpleVector<ValueType>> z;
    unsigned int z_row_size;

    s >> z_row_size;
    z.resize(z_row_size);
    for (unsigned int i = 0; i < z_row_size; i++) {
        unsigned int z_col_size;
        s >> z_col_size;
        for (unsigned int j = 0; j < z_col_size; j++) {
            ValueType temp;
            s >> temp;
            z[i].push_back(temp);
        }
    }

    aData.SetPreconditionedKrylovSpaceBasis(&z);

    s.PostStreamInAction(aData);
    return s;
}


template<typename ValueType, typename Stream>
Stream& operator<<(Stream& s, const KPreconditionedGeneralizedMinimalResidualState<ValueType>& aData)
{
    s.PreStreamOutAction(aData);

    s << static_cast<const KGeneralizedMinimalResidualState<ValueType>&>(aData);

    const std::vector<KSimpleVector<ValueType>>* z = aData.GetPreconditionedKrylovSpaceBasis();
    unsigned int z_row_size = z->size();
    s << z_row_size;

    for (unsigned int i = 0; i < z_row_size; i++) {
        unsigned int z_col_size = (z->at(i)).size();
        s << z_col_size;
        const KSimpleVector<ValueType>* temp = &(z->at(i));
        for (unsigned int j = 0; j < z_col_size; j++) {
            s << (*temp)(j);
        }
    }

    s.PostStreamOutAction(aData);

    return s;
}


}  // namespace KEMField

#endif /* __KPreconditionedGeneralizedMinimalResidualState_H__ */

#ifndef __KGeneralizedMinimalResidualState_H__
#define __KGeneralizedMinimalResidualState_H__

#include <string>
#include "KSimpleVector.hh"

#ifdef KEMFIELD_USE_MPI
    #include "KMPIInterface.hh"
    #ifndef MPI_ROOT_PROCESS_ONLY
        #define MPI_ROOT_PROCESS_ONLY if (KMPIInterface::GetInstance()->GetProcess()==0)
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
*@file KGeneralizedMinimalResidualState.hh
*@class KGeneralizedMinimalResidualState
*@brief
*@details
*<b>Revision History:<b>
*Date Name Brief Description
*Wed Jan 14 10:16:13 EST 2015 J. Barrett (barrettj@mit.edu) First Version
*
*/

template< typename ValueType >
class KGeneralizedMinimalResidualState
{
    public:

        typedef KSimpleVector<ValueType> KSimpleVectorType;

        KGeneralizedMinimalResidualState(){};
        virtual ~KGeneralizedMinimalResidualState(){};

        static std::string Name() { return std::string("gmres"); }
        std::string NameLabel() { return std::string("gmres"); }

        virtual void SetDimension(unsigned int dim){fDim = dim;};
        virtual unsigned int GetDimension() const {return fDim;};

        virtual void SetIterationCount(unsigned int j){fJ = j;};
        virtual unsigned int GetIterationCount() const {return fJ;};

        virtual void SetResidualNorm(ValueType res){fResidualNorm = res;};
        virtual ValueType GetResidualNorm() const {return fResidualNorm;};

        virtual const KSimpleVector<ValueType>* GetSolutionVector() const {return &fX;};
        virtual void SetSolutionVector(const KSimpleVector<ValueType>* x)
        {
            unsigned int size = x->size();
            fX.resize(size);
            for(unsigned int i=0; i<size; i++){fX[i] = (*x)(i);};
        };

        virtual const KSimpleVector<ValueType>* GetRightHandSide() const {return &fB;};
        virtual void SetRightHandSide(const KSimpleVector<ValueType>* b)
        {
            unsigned int size = b->size();
            fB.resize(size);
            for(unsigned int i=0; i<size; i++){fB[i] = (*b)(i);};
        };

        virtual const KSimpleVector<ValueType>* GetResidualVector() const {return &fR;};
        virtual void SetResidualVector(const KSimpleVector<ValueType>* r)
        {
            unsigned int size = r->size();
            fR.resize(size);
            for(unsigned int i=0; i<size; i++){fR[i] = (*r)(i);};
        };

        virtual const std::vector< KSimpleVector<ValueType> >* GetMinimizationMatrix() const { return &fH; };

        virtual void GetMinimizationMatrix(std::vector< KSimpleVector<ValueType> >* h) const
        {
            h->clear();
            unsigned int s = fH.size();
            h->resize(s);

            for(unsigned int i=0; i<s; i++)
            {
                unsigned int t = (fH.at(i)).size();
                (*h)[i].resize(t);
                for(unsigned int j = 0; j<t; j++)
                {
                    (*h)[i][j] = fH[i](j);
                }
            }
        }

        virtual void SetMinimizationMatrix(const std::vector< KSimpleVector<ValueType> >* h)
        {
            if(fH.size() != 0){for(unsigned int i=0; i<fH.size(); i++){fH[i].clear();};};
            fH.clear();

            unsigned int s = h->size();
            fH.resize(s);

            for(unsigned int i=0; i<s; i++)
            {
                unsigned int t = (h->at(i)).size();
                fH[i].resize(t);
                for(unsigned int j = 0; j<t; j++)
                {
                    fH[i][j] = (h->at(i))(j);
                }
            }
        };

        virtual const std::vector< KSimpleVector<ValueType> >* GetKrylovSpaceBasis() const {return &fV;};

        virtual void GetKrylovSpaceBasis(std::vector< KSimpleVector<ValueType> >* h) const
        {
            h->clear();
            unsigned int s = fV.size();
            h->resize(s);

            for(unsigned int i=0; i<s; i++)
            {
                unsigned int t = (fV.at(i)).size();
                (*h)[i].resize(t);
                for(unsigned int j = 0; j<t; j++)
                {
                    (*h)[i][j] = fV[i](j);
                }
            }
        }


        virtual void SetKrylovSpaceBasis(const std::vector< KSimpleVector<ValueType> >* v)
        {
            if(fV.size() != 0){for(unsigned int i=0; i<fV.size(); i++){fV[i].clear();};};
            fV.clear();

            unsigned int s = v->size();
            fV.resize(s);

            for(unsigned int i=0; i<s; i++)
            {
                unsigned int t = (v->at(i)).size();
                fV[i].resize(t);
                for(unsigned int j = 0; j<t; j++)
                {
                    fV[i][j] = (v->at(i))(j);
                }
            }
        };

        virtual const KSimpleVector<ValueType>* GetMinimizationRightHandSide() const {return &fP;};
        virtual void SetMinimizationRightHandSide(const KSimpleVector<ValueType>* p)
        {
            unsigned int size = p->size();
            fP.resize(size);
            for(unsigned int i=0; i<size; i++){fP[i] = (*p)(i);};
        };


        virtual const KSimpleVector<ValueType>* GetGivensRotationCosines() const {return &fC;};
        virtual void SetGivensRotationCosines(const KSimpleVector<ValueType>* c)
        {
            unsigned int size = c->size();
            fC.resize(size);
            for(unsigned int i=0; i<size; i++){fC[i] = (*c)(i);};
        };


        virtual const KSimpleVector<ValueType>* GetGivensRotationSines() const {return &fS;};
        virtual void SetGivensRotationSines(const KSimpleVector<ValueType>* s)
        {
            unsigned int size = s->size();
            fS.resize(size);
            for(unsigned int i=0; i<size; i++){fS[i] = (*s)(i);};
        };

        virtual void SynchronizeData()
        {

            //synchronize data across mpi process so that they all reflect the root process's state
            //IMPORTANT!! THIS ONLY WORKS FOR ValueType == double
            //Other types are not yet implemented for MPI version of this code
            #ifdef KEMFIELD_USE_MPI

            unsigned int x_size = fX.size();
            unsigned int b_size = fB.size();
            unsigned int r_size = fR.size();
            unsigned int h_row_size = fH.size();
            unsigned int v_row_size = fV.size();
            unsigned int p_size = fP.size();
            unsigned int c_size = fC.size();
            unsigned int sin_size = fS.size();

            const int root_id = 0;
            int proc_id = KMPIInterface::GetInstance()->GetProcess();

            KMPIInterface::GetInstance()->GlobalBarrier();

            MPI_Bcast( &(fDim), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(fJ), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(x_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(b_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(r_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(h_row_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(v_row_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(p_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(c_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(sin_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD );
            MPI_Bcast( &(fResidualNorm), 1, MPI_DOUBLE, root_id, MPI_COMM_WORLD );

            if(proc_id != root_id)
            {
                fX.resize(x_size);
                fB.resize(b_size);
                fR.resize(r_size);
                fH.resize(h_row_size);
                fV.resize(v_row_size);
                fP.resize(p_size);
                fC.resize(c_size);
                fS.resize(sin_size);
            }

            KMPIInterface::GetInstance()->GlobalBarrier();

            MPI_Bcast( &(fX[0]), x_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            MPI_Bcast( &(fB[0]), b_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            MPI_Bcast( &(fR[0]), r_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            MPI_Bcast( &(fP[0]), p_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            MPI_Bcast( &(fC[0]), c_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            MPI_Bcast( &(fS[0]), sin_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);

            for(unsigned int i=0; i<h_row_size; i++)
            {
                unsigned int h_col_size;
                if(proc_id == root_id)
                {
                    h_col_size = fH[i].size();
                }

                MPI_Bcast( &(h_col_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD);

                if(proc_id != root_id)
                {
                    fH[i].resize(h_col_size);
                }

                MPI_Bcast( &(fH[i][0]), h_col_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            }

/*
            for(unsigned int i=0; i<v_row_size; i++)
            {
                unsigned int v_col_size;
                if(proc_id == root_id)
                {
                    v_col_size = fV[i].size();
                }

                MPI_Bcast( &(v_col_size), 1, MPI_UNSIGNED, root_id, MPI_COMM_WORLD);

                if(proc_id != root_id)
                {
                    fV[i].resize(v_col_size);
                }

                MPI_Bcast( &(fV[i][0]), v_col_size, MPI_DOUBLE, root_id, MPI_COMM_WORLD);
            }
*/
            #endif

        }

    protected:

        //size of the system of equations
        unsigned int fDim;
        //inner iteration count
        unsigned int fJ;

        ValueType fResidualNorm;

        //solution vector
        KSimpleVector<ValueType> fX;
        //right hand side
        KSimpleVector<ValueType> fB;

        //intial residual vector
        KSimpleVector<ValueType> fR;

        //columns of H, the minimization matrix
        std::vector< KSimpleVector<ValueType> > fH;
        //the krylov subspace basis vectors
        std::vector< KSimpleVector<ValueType> > fV;

        KSimpleVector<ValueType> fP; //right hand side of minimization problem
        KSimpleVector<ValueType> fC; //Givens rotation cosines
        KSimpleVector<ValueType> fS; //Givens rotation sines
};



template <typename ValueType, typename Stream>
Stream& operator>>(Stream& s, KGeneralizedMinimalResidualState<ValueType>& aData)
{
    s.PreStreamInAction(aData);

    unsigned int dim;
    unsigned int iter;
    ValueType res_norm;
    KSimpleVector<ValueType> x;
    unsigned int x_size;
    KSimpleVector<ValueType> b;
    unsigned int b_size;
    KSimpleVector<ValueType> r;
    unsigned int r_size;
    std::vector< KSimpleVector<ValueType> > h;
    unsigned int h_row_size;
    std::vector< KSimpleVector<ValueType> > v;
    unsigned int v_row_size;
    KSimpleVector<ValueType> p;
    unsigned int p_size;
    KSimpleVector<ValueType> c;
    unsigned int c_size;
    KSimpleVector<ValueType> sin;
    unsigned int sin_size;

    s >> dim;
    s >> iter;
    s >> res_norm;

    s >> x_size;
    for(unsigned int i=0; i<x_size; i++)
    {
        ValueType temp;
        s >> temp;
        x.push_back(temp);
    }

    s >> b_size;
    for(unsigned int i=0; i<b_size; i++)
    {
        ValueType temp;
        s >> temp;
        b.push_back(temp);
    }

    s >> r_size;
    for(unsigned int i=0; i<r_size; i++)
    {
        ValueType temp;
        s >> temp;
        r.push_back(temp);
    }

    s >> h_row_size;
    h.resize(h_row_size);
    for(unsigned int i=0; i<h_row_size; i++)
    {
        unsigned int h_col_size;
        s >> h_col_size;
        for(unsigned int j=0; j<h_col_size; j++)
        {
            ValueType temp;
            s >> temp;
            h[i].push_back(temp);
        }
    }

    s >> v_row_size;
    v.resize(v_row_size);
    for(unsigned int i=0; i<v_row_size; i++)
    {
        unsigned int v_col_size;
        s >> v_col_size;
        for(unsigned int j=0; j<v_col_size; j++)
        {
            ValueType temp;
            s >> temp;
            v[i].push_back(temp);
        }
    }

    s >> p_size;
    for(unsigned int i=0; i<p_size; i++)
    {
        ValueType temp;
        s >> temp;
        p.push_back(temp);
    }

    s >> c_size;
    for(unsigned int i=0; i<c_size; i++)
    {
        ValueType temp;
        s >> temp;
        c.push_back(temp);
    }

    s >> sin_size;
    for(unsigned int i=0; i<sin_size; i++)
    {
        ValueType temp;
        s >> temp;
        sin.push_back(temp);
    }

    //now set this objects data
    aData.SetDimension(dim);
    aData.SetIterationCount(iter);
    aData.SetResidualNorm(res_norm);
    aData.SetSolutionVector(&x);
    aData.SetRightHandSide(&b);
    aData.SetResidualVector(&r);
    aData.SetMinimizationMatrix(&h);
    aData.SetKrylovSpaceBasis(&v);
    aData.SetMinimizationRightHandSide(&p);
    aData.SetGivensRotationCosines(&c);
    aData.SetGivensRotationSines(&sin);

    s.PostStreamInAction(aData);
    return s;
}




template <typename ValueType, typename Stream>
Stream& operator<<(Stream& s, const KGeneralizedMinimalResidualState<ValueType>& aData)
{
    s.PreStreamOutAction(aData);

    s << aData.GetDimension();
    s << aData.GetIterationCount();
    s << aData.GetResidualNorm();

    const KSimpleVector<ValueType>* x = aData.GetSolutionVector();
    unsigned int x_size = x->size();
    s << x_size;
    for(unsigned int i=0; i<x_size; i++)
    {
        s << (*x)(i);
    }

    const KSimpleVector<ValueType>* b = aData.GetRightHandSide();
    unsigned int b_size = b->size();
    s << b_size;
    for(unsigned int i=0; i<b_size; i++)
    {
        s << (*b)(i);
    }

    const KSimpleVector<ValueType>* r = aData.GetResidualVector();
    unsigned int r_size = r->size();
    s << r_size;
    for(unsigned int i=0; i<r_size; i++)
    {
        s << (*r)(i);
    }


    const std::vector< KSimpleVector<ValueType> >* h = aData.GetMinimizationMatrix();
    unsigned int h_row_size = h->size();
    s << h_row_size;

    for(unsigned int i=0; i<h_row_size; i++)
    {
        unsigned int h_col_size = (h->at(i)).size();
        s << h_col_size;
        const KSimpleVector<ValueType>* temp = &(h->at(i));
        for(unsigned int j=0; j<h_col_size; j++)
        {
            s << (*temp)(j);
        }
    }


    const std::vector< KSimpleVector<ValueType> >* v = aData.GetKrylovSpaceBasis();
    unsigned int v_row_size = v->size();
    s << v_row_size;

    for(unsigned int i=0; i<v_row_size; i++)
    {
        unsigned int v_col_size = (v->at(i)).size();
        s << v_col_size;
        const KSimpleVector<ValueType>* temp = &(v->at(i));
        for(unsigned int j=0; j<v_col_size; j++)
        {
            s << (*temp)(j);
        }
    }


    const KSimpleVector<ValueType>* p = aData.GetMinimizationRightHandSide();
    unsigned int p_size = p->size();
    s << p_size;
    for(unsigned int i=0; i<p_size; i++)
    {
        s << (*p)(i);
    }

    const KSimpleVector<ValueType>* c = aData.GetGivensRotationCosines();
    unsigned int c_size = c->size();
    s << c_size;
    for(unsigned int i=0; i<c_size; i++)
    {
        s << (*c)(i);
    }

    const KSimpleVector<ValueType>* sin = aData.GetGivensRotationSines();
    unsigned int sin_size = sin->size();
    s << sin_size;
    for(unsigned int i=0; i<sin_size; i++)
    {
        s << (*sin)(i);
    }


    s.PostStreamOutAction(aData);

    return s;
}





}

#endif /* __KGeneralizedMinimalResidualState_H__ */

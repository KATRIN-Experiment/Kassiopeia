#ifndef KROBINHOOD_MPI_DEF
#define KROBINHOOD_MPI_DEF

#include "KMPIInterface.hh"

namespace KEMField
{
  template <typename ValueType>
  class KRobinHood_MPI
  {
  public:
    typedef KSquareMatrix<ValueType> Matrix;
    typedef KVector<ValueType> Vector;

    KRobinHood_MPI(const Matrix& A,Vector& x,const Vector& b);
    ~KRobinHood_MPI();

    void Initialize();
    void FindResidual();
    void FindResidualNorm(double& residualNorm);
    void IdentifyLargestResidualElement();
    void ComputeCorrection();
    void UpdateSolutionApproximation();
    void UpdateVectorApproximation();
    void CoalesceData() {}
    void Finalize();

    unsigned int Dimension() const { return fB.Dimension(); }

    void SetResidualVector(const Vector&);
    void GetResidualVector(Vector&);

  private:
    const Matrix& fA;
    Vector& fX;
    const Vector& fB;

    KSimpleVector<ValueType> fB_iterative;
    KSimpleVector<ValueType> fResidual;

    double fBInfinityNorm;

    unsigned int fMaxResidualIndex;
    ValueType fMaxResidual;
    ValueType fMaxB_iterative;

    ValueType fCorrection;

    unsigned int fRank;
    unsigned int fNProcesses;

    unsigned int fOffset;
    unsigned int fRange;

    int* fOffsets;
    int* fRanges;

    inline unsigned int ToGlobal(unsigned int i) const { return i + fOffset; }
    inline unsigned int ToLocal(unsigned int i)  const { return i - fOffset; }

    MPI_Status   fStatus;
    MPI_Datatype fMPI_Res_type;
    MPI_Op       fMPI_Max;
    MPI_Op       fMPI_Min;

    typedef struct Res_Real_t {
      int    fIndex;
      double fRes;
      double fCorrection;
    } Res_Real;

    typedef struct Res_Complex_t {
      int    fIndex;
      double fRes;
      double fCorrection_real;
      double fCorrection_imag;
    } Res_Complex;

    Res_Real fRes_real;
    Res_Complex fRes_complex;

    static void MPIRealMax(Res_Real* in,
			   Res_Real* inout,
			   int* len,
			   MPI_Datatype*);
    static void MPIRealMin(Res_Real* in,
			   Res_Real* inout,
			   int* len,
			   MPI_Datatype*);
    static void MPIComplexMax(Res_Complex* in,
			      Res_Complex* inout,
			      int* len,
			      MPI_Datatype*);
    static void MPIComplexMin(Res_Complex* in,
			      Res_Complex* inout,
			      int* len,
			      MPI_Datatype*);

    void InitializeMPIStructs(Type2Type<double>);
    void InitializeMPIStructs(Type2Type<std::complex<double> >);

    void IdentifyLargestResidualElement(Type2Type<double>);
    void IdentifyLargestResidualElement(Type2Type<std::complex<double> >);

    void ComputeCorrection(Type2Type<double>);
    void ComputeCorrection(Type2Type<std::complex<double> >);

    void GetResidualVector(Type2Type<double>,Vector&);
    void GetResidualVector(Type2Type<std::complex<double> >,Vector&);
  };

  template <typename ValueType>
  KRobinHood_MPI<ValueType>::KRobinHood_MPI(const Matrix& A,Vector& x,const Vector& b) : fA(A), fX(x), fB(b)
  {
    fRank = KMPIInterface::GetInstance()->GetProcess();
    fNProcesses = KMPIInterface::GetInstance()->GetNProcesses();
    unsigned int remainder = fB.Dimension()%fNProcesses;

    fRange = fB.Dimension()/fNProcesses + (fRank<remainder ? 1 : 0);
    fOffset = (fRank*(fB.Dimension()/fNProcesses) +
	       (fRank<remainder ? fRank : remainder));

    fRanges = new int[fNProcesses];
    fOffsets = new int[fNProcesses];

    for (unsigned int i=0;i<fNProcesses;i++)
    {
      fRanges[i] = fB.Dimension()/fNProcesses + (i<remainder ? 1 : 0);
      fOffsets[i] = (i*(fB.Dimension()/fNProcesses) +
		     (i<remainder ? i : remainder));
    }

    InitializeMPIStructs(Type2Type<ValueType>());
  }

  template <typename ValueType>
  KRobinHood_MPI<ValueType>::~KRobinHood_MPI()
  {
    if (fMPI_Max != MPI_OP_NULL)
      MPI_Op_free(&fMPI_Max);
    if (fMPI_Min != MPI_OP_NULL)
      MPI_Op_free(&fMPI_Min);

    delete [] fRanges;
    delete [] fOffsets;
  }
  
  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::Initialize()
  {
    if (fResidual.Dimension()==0)
    {
      fB_iterative.resize(fRange,0.);
      fResidual.resize(fRange,0.);

      if (fX.InfinityNorm()>1.e-16)
	for (unsigned int i=0;i<fRange;i++)
	  for (unsigned int j=0;j<fX.Dimension();j++)
	    fB_iterative[i] += fA(ToGlobal(i),j)*fX(j);
    }

    fBInfinityNorm = fB.InfinityNorm();

    MPI_Barrier(MPI_COMM_WORLD);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::FindResidual()
  {
    for (unsigned int i=0;i<fResidual.Dimension();i++)
      fResidual[i] = fB(ToGlobal(i)) - fB_iterative(i);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::FindResidualNorm(double& residualNorm)
  {
    residualNorm = 0.;
    for (unsigned int i=0;i<fResidual.Dimension();i++)
      if (fabs(fResidual(i))>residualNorm)
	residualNorm = fabs(fResidual(i));

    MPI_Allreduce(MPI_IN_PLACE,&residualNorm,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    residualNorm/=fBInfinityNorm;
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::IdentifyLargestResidualElement()
  {
    fMaxResidualIndex = 0;
    fMaxResidual = fabs(fResidual(0));
    for (unsigned int i=0;i<fResidual.Dimension();i++)
    {
      if (fabs(fResidual(i))>fMaxResidual)
      {
	fMaxResidual = fabs(fResidual(i));
	fMaxResidualIndex = i;
      }
    }
    fMaxB_iterative = fB_iterative(fMaxResidualIndex);
    fMaxResidualIndex = ToGlobal(fMaxResidualIndex);
    IdentifyLargestResidualElement(Type2Type<ValueType>());
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::IdentifyLargestResidualElement(Type2Type<double>)
  {
    fRes_real.fIndex = fMaxResidualIndex;
    fRes_real.fRes = fabs(fMaxResidual);
    fRes_real.fCorrection = (fB(fMaxResidualIndex) - fMaxB_iterative)/fA(fMaxResidualIndex,fMaxResidualIndex);

    MPI_Allreduce(MPI_IN_PLACE,&fRes_real,1,fMPI_Res_type,fMPI_Max,MPI_COMM_WORLD);

    fMaxResidualIndex = fRes_real.fIndex;
    fCorrection = fRes_real.fCorrection;
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::IdentifyLargestResidualElement(Type2Type<std::complex<double> >)
  {
    fRes_complex.fIndex = fMaxResidualIndex;
    fRes_complex.fRes = fabs(fMaxResidual);
    fRes_complex.fCorrection_real = ((fB(fMaxResidualIndex) - fMaxB_iterative)/fA(fMaxResidualIndex,fMaxResidualIndex)).real();
    fRes_complex.fCorrection_imag = ((fB(fMaxResidualIndex) - fMaxB_iterative)/fA(fMaxResidualIndex,fMaxResidualIndex)).imag();

    MPI_Allreduce(MPI_IN_PLACE,&fRes_complex,1,fMPI_Res_type,fMPI_Max,MPI_COMM_WORLD);

    fMaxResidualIndex = fRes_complex.fIndex;
    fCorrection = std::complex<double>(fRes_complex.fCorrection_real,fRes_complex.fCorrection_imag);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::ComputeCorrection()
  {
    ComputeCorrection(Type2Type<ValueType>());
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::ComputeCorrection(Type2Type<double>)
  {
    fCorrection = fRes_real.fCorrection;
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::ComputeCorrection(Type2Type<std::complex<double> >)
  {
    fCorrection = std::complex<double>(fRes_complex.fCorrection_real,
					 fRes_complex.fCorrection_imag);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::UpdateSolutionApproximation()
  {
    fX[fMaxResidualIndex] += fCorrection;
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::UpdateVectorApproximation()
  {
    for (unsigned int i = 0;i<fB_iterative.Dimension();i++)
      fB_iterative[i] += fA(ToGlobal(i),fMaxResidualIndex)*fCorrection;
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::Finalize()
  {
    MPI_Barrier(MPI_COMM_WORLD);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::SetResidualVector(const Vector& v)
  {
    fResidual.resize(fRange);
    fB_iterative.resize(fRange);

    for (unsigned int i=0;i<fResidual.Dimension();i++)
    {
      fResidual[i] = v(ToGlobal(i));
      fB_iterative[i] = fB(ToGlobal(i)) - fResidual(i);
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::GetResidualVector(Vector& v)
  {
    GetResidualVector(Type2Type<ValueType>(),v);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::GetResidualVector(Type2Type<double>,Vector& v)
  {
    MPI_Gatherv(&fResidual[0],fResidual.Dimension(),MPI_DOUBLE,&v[0],&fRanges[0],&fOffsets[0],MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::GetResidualVector(Type2Type<std::complex<double> >,Vector& v)
  {
    std::vector<double> residual_real(fResidual.Dimension());
    std::vector<double> residual_imag(fResidual.Dimension());

    std::vector<double> v_real;
    std::vector<double> v_imag;

    if (fRank == 0)
    {
      v_real.resize(v.Dimension());
      v_imag.resize(v.Dimension());
    }

    for (unsigned int i=0;i<fResidual.Dimension();i++)
    {
      residual_real[i] = fResidual[i].real();
      residual_imag[i] = fResidual[i].imag();
    }

    MPI_Gatherv(&residual_real[0],fResidual.Dimension(),MPI_DOUBLE,&v_real[0],&fRanges[0],&fOffsets[0],MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gatherv(&residual_imag[0],fResidual.Dimension(),MPI_DOUBLE,&v_imag[0],&fRanges[0],&fOffsets[0],MPI_DOUBLE,0,MPI_COMM_WORLD);

    if (fRank == 0)
    {
      for (unsigned int i=0;i<v.Dimension();i++)
	v[i] = std::complex<double>(v_real[i],v_imag[i]);
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::MPIRealMax(Res_Real* in,
					 Res_Real* inout,
					 int* len,
					 MPI_Datatype*) 
  { 
    int i; 
 
    for (i=0; i< *len; ++i)
    { 
      if (in->fRes > inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex = in->fIndex; 
	inout->fRes  = in->fRes;
	inout->fCorrection  = in->fCorrection;
      }
      in++; inout++; 
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::MPIRealMin(Res_Real* in,
					 Res_Real* inout,
					 int* len,
					 MPI_Datatype*) 
  { 
    int i; 
 
    for (i=0; i< *len; ++i)
    { 
      if (in->fRes < inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex = in->fIndex; 
	inout->fRes  = in->fRes;
	inout->fCorrection  = in->fCorrection;
      }
      in++; inout++; 
    }
  } 

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::MPIComplexMax(Res_Complex* in,
					    Res_Complex* inout,
					    int* len,
					    MPI_Datatype*) 
  { 
    int i; 
 
    for (i=0; i< *len; ++i)
    { 
      if (in->fRes > inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex  = in->fIndex; 
	inout->fRes = in->fRes;
	inout->fCorrection_real = in->fCorrection_real;
	inout->fCorrection_imag = in->fCorrection_imag;
      }
      in++; inout++; 
    }
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::MPIComplexMin(Res_Complex* in,
					    Res_Complex* inout,
					    int* len,
					    MPI_Datatype*) 
  { 
    int i; 
 
    for (i=0; i< *len; ++i)
    { 
      if (in->fRes < inout->fRes ||
	  (in->fRes == inout->fRes && in->fIndex < inout->fIndex))
      {
	inout->fIndex  = in->fIndex; 
	inout->fRes = in->fRes;
	inout->fCorrection_real = in->fCorrection_real;
	inout->fCorrection_imag = in->fCorrection_imag;
      }
      in++; inout++; 
    }
  } 

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::InitializeMPIStructs(Type2Type<double>)
  {
    int block_lengths[3] = {1,1,1};
    MPI_Aint displacements[3];
    MPI_Aint addresses[4];
    MPI_Datatype typelist[3] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE};
    
    MPI_Get_address(&fRes_real,&addresses[0]);
    MPI_Get_address(&(fRes_real.fIndex),&addresses[1]);
    MPI_Get_address(&(fRes_real.fRes),&addresses[2]);
    MPI_Get_address(&(fRes_real.fCorrection),&addresses[3]);

    displacements[0] = addresses[1] - addresses[0];
    displacements[1] = addresses[2] - addresses[0];
    displacements[2] = addresses[3] - addresses[0];

    MPI_Type_create_struct(3,
  			   block_lengths,
  			   displacements,
  			   typelist,
  			   &fMPI_Res_type);

    MPI_Type_commit(&fMPI_Res_type);

    MPI_Op_create((MPI_User_function *)MPIRealMax,true,&fMPI_Max);
    MPI_Op_create((MPI_User_function *)MPIRealMin,true,&fMPI_Min);
  }

  template <typename ValueType>
  void KRobinHood_MPI<ValueType>::InitializeMPIStructs(Type2Type<std::complex<double> >)
  {
    int block_lengths[4] = {1,1,1,1};
    MPI_Aint displacements[4];
    MPI_Aint addresses[5];
    MPI_Datatype typelist[4] = {MPI_INT,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
    
    MPI_Get_address(&fRes_complex,&addresses[0]);
    MPI_Get_address(&(fRes_complex.fIndex),&addresses[1]);
    MPI_Get_address(&(fRes_complex.fRes),&addresses[2]);
    MPI_Get_address(&(fRes_complex.fCorrection_real),&addresses[3]);
    MPI_Get_address(&(fRes_complex.fCorrection_imag),&addresses[4]);

    displacements[0] = addresses[1] - addresses[0];
    displacements[1] = addresses[2] - addresses[0];
    displacements[2] = addresses[3] - addresses[0];
    displacements[3] = addresses[4] - addresses[0];

    MPI_Type_create_struct(4,
  			   block_lengths,
  			   displacements,
  			   typelist,
  			   &fMPI_Res_type);

    MPI_Type_commit(&fMPI_Res_type);

    MPI_Op_create((MPI_User_function *)MPIComplexMax,true,&fMPI_Max);
    MPI_Op_create((MPI_User_function *)MPIComplexMin,true,&fMPI_Min);
  }
}

#endif /* KROBINHOOD_MPI_DEF */

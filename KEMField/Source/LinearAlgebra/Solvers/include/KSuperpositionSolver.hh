#ifndef KSUPERPOSITIONSOLVER_DEF
#define KSUPERPOSITIONSOLVER_DEF

#include "KEMCout.hh"
#include "KSimpleMatrix.hh"
#include "KSimpleVector.hh"

namespace KEMField
{
template<typename ValueType, template<typename> class SVDSolver> class KSuperpositionSolver
{
  public:
    typedef KVector<ValueType> Vector;

    KSuperpositionSolver() : fTolerance(1.e-14) {}
    virtual ~KSuperpositionSolver() {}

    void SetTolerance(double d)
    {
        fTolerance = d;
    }

    void AddSolvedSystem(const Vector& x, const Vector& b);

    void Solve(Vector& x, const Vector& b);

    bool SolutionSpaceIsSpanned(const Vector& b) const;
    void ComposeSolution(Vector& x) const;

  private:
    bool Reduce(const Vector& b) const;
    bool ComputeWeights() const;

    class Matrix : public KMatrix<ValueType>
    {
      public:
        Matrix(std::vector<std::vector<ValueType>>& elements) : fElements(elements) {}
        ~Matrix() override {}

        unsigned int Dimension(unsigned int) const override;
        const ValueType& operator()(unsigned int, unsigned int) const override;

      private:
        std::vector<std::vector<ValueType>>& fElements;
    };

    std::vector<const Vector*> fX;
    mutable std::vector<const Vector*> fB;
    mutable std::vector<std::vector<ValueType>> fReducedB;
    mutable KSimpleVector<ValueType> fWeights;

    double fTolerance;
};

template<typename ValueType, template<typename> class SVDSolver>
unsigned int KSuperpositionSolver<ValueType, SVDSolver>::Matrix::Dimension(unsigned int i) const
{
    if (i == 0)
        return fElements.at(0).size();
    else
        return fElements.size();
}

template<typename ValueType, template<typename> class SVDSolver>
const ValueType& KSuperpositionSolver<ValueType, SVDSolver>::Matrix::operator()(unsigned int i, unsigned int j) const
{
    return fElements.at(j).at(i);
}

template<typename ValueType, template<typename> class SVDSolver>
void KSuperpositionSolver<ValueType, SVDSolver>::AddSolvedSystem(const Vector& x, const Vector& b)
{
    fX.push_back(&x);
    fB.push_back(&b);
}

template<typename ValueType, template<typename> class SVDSolver>
bool KSuperpositionSolver<ValueType, SVDSolver>::SolutionSpaceIsSpanned(const Vector& b) const
{
    if (Reduce(b))
        return ComputeWeights();
    return false;
}

template<typename ValueType, template<typename> class SVDSolver>
bool KSuperpositionSolver<ValueType, SVDSolver>::Reduce(const Vector& b) const
{
    // Reduces our space to the number of independent parameters in our system.
    // To do this, we do an element-wise comparison of the precomputed vectors
    // and determine a common index across them.  Returns false if the prior
    // systems do not adequately span the solution space, and true if they do.

    if (fB.empty())
        return false;

    fB.push_back(&b);

    fReducedB.resize(fB.size());

    unsigned int dimension = b.Dimension();

    // loop over elements
    for (unsigned int i = 0; i < dimension; i++) {
        unsigned int commonIndex = 0;
        bool indexFound = false;

        while (!indexFound) {
            indexFound = true;

            // if <commonIndex> exceeds the dimensions of our reduced space, then we
            // extend our reduced space
            if (commonIndex == fReducedB[0].size())
                for (unsigned int j = 0; j < fB.size(); j++)
                    fReducedB[j].push_back((*fB[j])(i));

            // loop over solved systems
            for (unsigned int j = 0; j < fB.size(); j++) {
                // if element <i> cannot be represented by reduced element <k>...
                if (fabs(fReducedB[j][commonIndex] - (*fB[j])(i)) > fTolerance) {
                    // ...if the problematic vector is the vector in the equation we are
                    // trying to solve, then our prior solutions do not span the
                    // solution space of interest.
                    if (j == fB.size())
                        return false;

                    // Otherwise, increment the common index, and restart the search.
                    commonIndex++;
                    indexFound = false;
                    break;
                }
            }
        }
    }

    return true;
}

template<typename ValueType, template<typename> class SVDSolver>
bool KSuperpositionSolver<ValueType, SVDSolver>::ComputeWeights() const
{
    // Solves our reduced system to determine the weighting factors that must be
    // assigned to our prior solution vectors in order to construct the new
    // solution.  Returns false if the prior systems do not adequately span the
    // solution space, and true if they do.

    KSimpleVector<ValueType> b(fReducedB.back());
    fReducedB.pop_back();
    Matrix A(fReducedB);

    fWeights.resize(A.Dimension(1));
    SVDSolver<ValueType> solver;
    return solver.Solve(A, fWeights, b);
}

template<typename ValueType, template<typename> class SVDSolver>
void KSuperpositionSolver<ValueType, SVDSolver>::ComposeSolution(Vector& x) const
{
    // Performs a weighted sum of the prior solutions to construct the new
    // solution vector.

    for (unsigned int i = 0; i < x.Dimension(); i++)
        x[i] = 0;

    for (unsigned int i = 0; i < fX.size(); i++) {
        KSimpleVector<ValueType> tmp(x.Dimension());
        fX[i]->Multiply(fWeights[i], tmp);
        x += tmp;
    }
}
}  // namespace KEMField

#endif /* KSUPERPOSITIONSOLVER_DEF */

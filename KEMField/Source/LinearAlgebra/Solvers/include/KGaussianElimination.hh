#ifndef KGAUSSIANELIMINATION_DEF
#define KGAUSSIANELIMINATION_DEF

#include "KSquareMatrix.hh"
#include "KVector.hh"

namespace KEMField
{
template<typename ValueType> class KGaussianElimination
{
  public:
    using Matrix = KSquareMatrix<ValueType>;
    using Vector = KVector<ValueType>;

    KGaussianElimination() = default;
    virtual ~KGaussianElimination() = default;

    void Solve(const Matrix& A, Vector& x, const Vector& b) const;

  protected:
};

template<typename ValueType>
void KGaussianElimination<ValueType>::Solve(const Matrix& A, Vector& x, const Vector& b) const
{
    int i, j;

    int n = A.Dimension();
    auto** a = new double*[n];
    for (i = 0; i < n; i++)
        a[i] = new double[n + 1];

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            a[i][j] = A(i, j);

    for (j = 0; j < n; j++)
        a[j][n] = b(j);

    int row, col;

    /* gaussian elimination */
    for (col = 0; col < n; col++) {
        for (row = 0; row < n; row++) {
            double pivot = a[row][col] / a[col][col];
            if (row != col) {
                for (i = 0; i < n + 1; i++) {
                    a[row][i] = a[row][i] - pivot * a[col][i];
                }
            }
        }
    }

    /* X = B/A and show X */
    for (row = 0; row < n; row++) {
        x[row] = a[row][n] / a[row][row];
    }
}
}  // namespace KEMField

#endif /* KGAUSSIANELIMINATION_DEF */

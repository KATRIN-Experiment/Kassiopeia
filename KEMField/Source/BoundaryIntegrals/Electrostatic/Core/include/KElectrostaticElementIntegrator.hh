/*
 * KElectrostaticElementIntegrator.hh
 *
 *  Created on: 25 Aug 2016
 *      Author: wolfgang
 */

#ifndef KELECTROSTATICELEMENTINTEGRATOR_HH_
#define KELECTROSTATICELEMENTINTEGRATOR_HH_

#include "KSymmetryGroup.hh"
#include "KThreeVector_KEMField.hh"

namespace KEMField
{

template<class Shape> class KElectrostaticElementIntegrator
{
  public:
    virtual ~KElectrostaticElementIntegrator(){};
    virtual double Potential(const Shape* source, const KPosition& P) const = 0;
    virtual KThreeVector ElectricField(const Shape* source, const KPosition& P) const = 0;
    virtual std::pair<KThreeVector, double> ElectricFieldAndPotential(const Shape* source, const KPosition& P) const;

    virtual double Potential(const KSymmetryGroup<Shape>* source, const KPosition& P) const;
    virtual KThreeVector ElectricField(const KSymmetryGroup<Shape>* source, const KPosition& P) const;
    virtual std::pair<KThreeVector, double> ElectricFieldAndPotential(const KSymmetryGroup<Shape>* source,
                                                                      const KPosition& P) const;

  private:
    typedef typename KSymmetryGroup<Shape>::ShapeCIt ShapeGroupCIt;
};


template<class Shape>
inline std::pair<KThreeVector, double>
KElectrostaticElementIntegrator<Shape>::ElectricFieldAndPotential(const Shape* source, const KPosition& P) const
{
    return std::make_pair(ElectricField(source, P), Potential(source, P));
}

template<class Shape>
inline double KElectrostaticElementIntegrator<Shape>::Potential(const KSymmetryGroup<Shape>* source,
                                                                const KPosition& P) const
{
    double potential = 0.;
    for (auto it = source->begin(); it != source->end(); ++it)
        potential += Potential(*it, P);
    return potential;
}

template<class Shape>
inline KThreeVector KElectrostaticElementIntegrator<Shape>::ElectricField(const KSymmetryGroup<Shape>* source,
                                                                          const KPosition& P) const
{
    KThreeVector electricField(0., 0., 0.);
    for (auto it = source->begin(); it != source->end(); ++it)
        electricField += ElectricField(*it, P);
    return electricField;
}

template<class Shape>
inline std::pair<KThreeVector, double>
KElectrostaticElementIntegrator<Shape>::ElectricFieldAndPotential(const KSymmetryGroup<Shape>* source,
                                                                  const KPosition& P) const
{
    std::pair<KThreeVector, double> fieldAndPotential;
    double potential(0.);
    KThreeVector electricField(0., 0., 0.);

    for (auto it = source->begin(); it != source->end(); ++it) {
        fieldAndPotential = ElectricFieldAndPotential(*it, P);
        electricField += fieldAndPotential.first;
        potential += fieldAndPotential.second;
    }

    return std::make_pair(electricField, potential);
}

} /* namespace KEMField */

#endif /* KELECTROSTATICELEMENTINTEGRATOR_HH_ */

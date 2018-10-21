/*
 * KElectrostaticElementIntegrator.hh
 *
 *  Created on: 25 Aug 2016
 *      Author: wolfgang
 */

#ifndef KEMFIELD_SOURCE_2_0_BOUNDARYINTEGRALS_ELECTROSTATIC_INCLUDE_KELECTROSTATICELEMENTINTEGRATOR_HH_
#define KEMFIELD_SOURCE_2_0_BOUNDARYINTEGRALS_ELECTROSTATIC_INCLUDE_KELECTROSTATICELEMENTINTEGRATOR_HH_


namespace KEMField {

template< class Shape>
class KElectrostaticElementIntegrator
{
public:
    virtual ~KElectrostaticElementIntegrator(){};
    virtual double Potential( const Shape* source, const KPosition& P ) const = 0;
    virtual KEMThreeVector ElectricField( const Shape* source, const KPosition& P) const = 0;
    virtual std::pair<KEMThreeVector,double> ElectricFieldAndPotential( const Shape* source, const KPosition& P) const;

    virtual double Potential( const KSymmetryGroup<Shape>* source, const KPosition& P ) const;
    virtual KEMThreeVector ElectricField( const KSymmetryGroup<Shape>* source, const KPosition& P ) const;
    virtual std::pair<KEMThreeVector,double> ElectricFieldAndPotential( const KSymmetryGroup<Shape>* source, const KPosition& P) const;
private:
    typedef typename KSymmetryGroup<Shape>::ShapeCIt ShapeGroupCIt;
};





template<class Shape>
inline std::pair<KEMThreeVector, double> KElectrostaticElementIntegrator<
Shape>::ElectricFieldAndPotential(const Shape* source,
        const KPosition& P) const
{
    return std::make_pair( ElectricField(source,P), Potential(source,P) );
}

template<class Shape>
inline double KElectrostaticElementIntegrator<Shape>::Potential(
        const KSymmetryGroup<Shape>* source, const KPosition& P) const
{
    double potential = 0.;
    for (ShapeGroupCIt it=source->begin();it!=source->end();++it)
        potential += Potential(*it,P);
    return potential;
}

template<class Shape>
inline KEMThreeVector KElectrostaticElementIntegrator<Shape>::ElectricField(
        const KSymmetryGroup<Shape>* source, const KPosition& P) const
{
    KEMThreeVector electricField(0.,0.,0.);
    for ( ShapeGroupCIt it=source->begin();it!=source->end();++it)
        electricField += ElectricField(*it,P);
    return electricField;
}

template<class Shape>
inline std::pair<KEMThreeVector, double> KElectrostaticElementIntegrator<
Shape>::ElectricFieldAndPotential(
        const KSymmetryGroup<Shape>* source, const KPosition& P) const
{
    std::pair<KEMThreeVector, double> fieldAndPotential;
    double potential( 0. );
    KEMThreeVector electricField( 0., 0., 0. );

    for( ShapeGroupCIt it=source->begin(); it!=source->end(); ++it )
    {
        fieldAndPotential = ElectricFieldAndPotential( *it, P );
        electricField += fieldAndPotential.first;
        potential += fieldAndPotential.second;
    }

    return std::make_pair( electricField, potential );
}

} /* namespace KEMField */

#endif /* KEMFIELD_SOURCE_2_0_BOUNDARYINTEGRALS_ELECTROSTATIC_INCLUDE_KELECTROSTATICELEMENTINTEGRATOR_HH_ */

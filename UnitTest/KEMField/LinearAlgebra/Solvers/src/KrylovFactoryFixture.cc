/*
 * KrylovFactoryFixture.cc
 *
 *  Created on: 13 Aug 2015
 *      Author: wolfgang
 */

#include "KrylovFactoryFixture.hh"

KrylovFactoryFixture::KrylovFactoryFixture() :
    fA(new KEMField::KSimpleSquareMatrix<ElectricType>(1)),
    fP(new KEMField::KSimpleSquareMatrix<ElectricType>(1))
{}

KrylovFactoryFixture::~KrylovFactoryFixture()
{
    // TODO Auto-generated destructor stub
}

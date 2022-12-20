#!/bin/sh

VERSION="${1:-testing}"
IMAGE="katrinexperiment/kassiopeia"

RELEASE="$(git describe --tags --exact-match || git symbolic-ref -q --short HEAD)"

if [ "${VERSION}" != "${RELEASE}" -a "v${VERSION}" != "${RELEASE}" ]; then
    echo "ERROR: Provided image version $VERSION does not match git branch/tag: $RELEASE"
    exit 1
fi

shift 1

echo "-- building $IMAGE:$VERSION ..."
sudo docker build -t $IMAGE $@ . || exit $?
sudo docker run -it katrinexperiment/kassiopeia UnitTestKasper || exit $?

######
#exit 0
######

echo "-- pushing to DockerHub ..."
sudo docker tag $IMAGE:latest katrinexperiment/kassiopeia:$VERSION
sudo docker push katrinexperiment/kassiopeia:$VERSION
sudo docker push katrinexperiment/kassiopeia:latest

echo "-- pushing to GitHub ..."
sudo docker tag $IMAGE:latest docker.pkg.github.com/katrin-experiment/kassiopeia/kassiopeia:$VERSION
sudo docker push docker.pkg.github.com/katrin-experiment/kassiopeia/kassiopeia:$VERSION

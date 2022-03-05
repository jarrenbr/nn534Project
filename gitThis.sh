#!/bin/bash

find . -type f -size +99M

echo "Proceed with the above files above 99 Megabytes? Y/n"

read proceed

if [ "$proceed" == "n" ] || [ "$proceed" == "N" ]
then
	exit 1
fi

git add .
if [ "$#" -eq 1 ]
then
	git commit -m "$1";
else
	git commit -m "default message";
fi

git push;

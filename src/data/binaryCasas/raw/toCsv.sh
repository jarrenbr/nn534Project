#!/bin/bash

for f in b1.test b1.train b2.test b2.train b3.test b3.train; do
	echo -e "Date Time Sensor Signal Activity" > ${f}.csv ; cat $f >> ${f}.csv; sed -i 's/ /,/g' ${f}.csv;
done


